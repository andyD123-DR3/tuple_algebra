#ifndef CTDP_CALIBRATOR_FIX_STRATEGY_IDS_H
#define CTDP_CALIBRATOR_FIX_STRATEGY_IDS_H

// ============================================================
//  fix_strategy_ids.h  –  CT-DP FIX Parser Optimiser
//
//  Dense schema-aware plan indexing.
//
//  Problem with raw plan_id (base-4):
//    trivial_schema has 4×4×1×4 = 64 valid plans but plan_id
//    addresses a 4^4 = 256 entry table.  192 slots are wasted,
//    and the DP cost table would be sparsely populated.
//
//  Solution: SchemaIndex<N>
//    Mixed-radix encoding where the radix of field i is
//    n_allowed[i] (not always 4).  This gives a dense index
//    in [0, plan_space_size) with no gaps.
//
//    dense_id = Σ local_id[i] * stride[i]
//
//    where:
//      local_id[i]  = position of plan[i] within the sorted
//                     allowed-strategy list for field i
//                     (U < S < L < G, matching Strategy enum order)
//      stride[0]    = 1
//      stride[i]    = Π n_allowed[j] for j < i
//
//  Example — trivial_schema:
//    Field        n_allowed  stride
//    BeginString  4          1
//    BodyLength   4          4
//    MsgType      1          16   ← locked; always contributes 0
//    CheckSum     4          16
//
//    Plan "SUSG" → local_ids [1,0,0,3] → 1 + 0 + 0 + 48 = 49
//
//  StrategyLocalMap<N>:
//    Per-field lookup table: Strategy → local_id (or -1 if not allowed).
//    Built at compile time from the Schema<N>.
//
//  Key operations (all constexpr):
//    encode(plan) → dense_id
//    decode(dense_id) → plan
//    is_valid_id(dense_id) → bool
//    local_id(field, strategy) → int  (-1 if not allowed)
//    local_to_strategy(field, local_id) → Strategy
//
//  DP table helper:
//    CostTable<N, T>: flat array of plan_space_size() entries of type T,
//    indexed by dense_id.  T is typically double (measured latency) or
//    a user-defined cost struct.
//
//  Dependencies: fix_field_descriptor.h
//  C++ standard: C++20
// ============================================================

#include <ctdp/calibrator/fix/fix_field_descriptor.h>

#include <array>
#include <cassert>
#include <cstdint>
#include <limits>
#include <optional>
#include <stdexcept>
#include <vector>

namespace ctdp::calibrator::fix {

// ─────────────────────────────────────────────────────────────────────────────
//  StrategyLocalMap — per-field mapping: Strategy → local_id within allowed set
//
//  allowed_strategies is sorted by Strategy enum value (U=0,S=1,L=2,G=3)
//  so the local_id is simply the position in that sorted list.
//
//  to_local[s] = local position of Strategy s for this field (-1 if not allowed)
//  from_local[k] = Strategy at local position k
// ─────────────────────────────────────────────────────────────────────────────

struct StrategyLocalMap {
    // to_local[strategy_index] → local_id, or -1 if not allowed
    std::array<int8_t, NUM_STRATEGIES> to_local{};
    // from_local[local_id] → Strategy
    std::array<Strategy, NUM_STRATEGIES> from_local{};
    int n_allowed{0};

    constexpr StrategyLocalMap() noexcept {
        to_local.fill(-1);
    }

    constexpr void build(AllowedMask mask) noexcept {
        int local = 0;
        for (int si = 0; si < NUM_STRATEGIES; ++si) {
            if ((mask >> si) & 1u) {
                to_local[si] = static_cast<int8_t>(local);
                from_local[local] = static_cast<Strategy>(si);
                ++local;
            }
        }
        n_allowed = local;
    }

    [[nodiscard]] constexpr int to(Strategy s) const noexcept {
        return to_local[static_cast<int>(s)];
    }

    [[nodiscard]] constexpr Strategy from(int local) const noexcept {
        assert(local >= 0 && local < n_allowed);
        return from_local[local];
    }

    [[nodiscard]] constexpr bool allows(Strategy s) const noexcept {
        return to_local[static_cast<int>(s)] >= 0;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  SchemaIndex<N> — dense mixed-radix encode/decode for a Schema<N>
// ─────────────────────────────────────────────────────────────────────────────

template<int N>
    requires (N >= 1 && N <= 16)
struct SchemaIndex {
    std::array<StrategyLocalMap, N> local_maps{};
    std::array<int32_t, N>          strides{};
    int32_t                         total{0};

    constexpr SchemaIndex() = default;

    explicit constexpr SchemaIndex(const Schema<N>& schema) noexcept {
        // Build per-field local maps
        for (int i = 0; i < N; ++i)
            local_maps[i].build(schema.fields[i].allowed_mask);

        // Compute mixed-radix strides: stride[0]=1, stride[i]=stride[i-1]*n_allowed[i-1]
        strides[0] = 1;
        for (int i = 1; i < N; ++i)
            strides[i] = strides[i-1] * local_maps[i-1].n_allowed;

        total = strides[N-1] * local_maps[N-1].n_allowed;
    }

    // ── Core encode/decode ───────────────────────────────────────────────────

    // plan → dense_id in [0, total).  Returns -1 if plan contains a
    // disallowed strategy for any field.
    [[nodiscard]] constexpr int32_t encode(
        const std::array<Strategy, N>& plan) const noexcept
    {
        int32_t id = 0;
        for (int i = 0; i < N; ++i) {
            int local = local_maps[i].to(plan[i]);
            if (local < 0) return -1;  // disallowed strategy
            id += local * strides[i];
        }
        return id;
    }

    // dense_id → plan.  UB if dense_id is out of range (assert-guarded).
    [[nodiscard]] constexpr std::array<Strategy, N>
    decode(int32_t dense_id) const noexcept {
        assert(dense_id >= 0 && dense_id < total);
        std::array<Strategy, N> plan{};
        int32_t rem = dense_id;
        // decode from highest field downward to avoid modular arithmetic
        for (int i = N-1; i >= 0; --i) {
            int local = static_cast<int>(rem / strides[i]);
            rem       = rem % strides[i];
            plan[i]   = local_maps[i].from(local);
        }
        return plan;
    }

    // ── Accessors ────────────────────────────────────────────────────────────

    [[nodiscard]] constexpr int32_t size() const noexcept { return total; }

    [[nodiscard]] constexpr bool is_valid_id(int32_t id) const noexcept {
        return id >= 0 && id < total;
    }

    // Local id of strategy s at field fi (-1 if not allowed)
    [[nodiscard]] constexpr int local_id(int fi, Strategy s) const noexcept {
        assert(fi >= 0 && fi < N);
        return local_maps[fi].to(s);
    }

    // Strategy at local position k for field fi
    [[nodiscard]] constexpr Strategy local_to_strategy(int fi, int k) const noexcept {
        assert(fi >= 0 && fi < N);
        return local_maps[fi].from(k);
    }

    // Number of allowed strategies at field fi
    [[nodiscard]] constexpr int n_allowed(int fi) const noexcept {
        assert(fi >= 0 && fi < N);
        return local_maps[fi].n_allowed;
    }

    // ── Neighbour iteration (for DP / beam search) ───────────────────────────
    //
    // Neighbours of dense_id in dimension (field) fi: all plans identical
    // to decode(dense_id) except plan[fi] varies over allowed strategies.
    // Returns the dense_ids of those neighbours (including dense_id itself).

    [[nodiscard]] std::vector<int32_t>
    field_neighbours(int32_t dense_id, int fi) const {
        assert(is_valid_id(dense_id));
        assert(fi >= 0 && fi < N);
        std::vector<int32_t> result;
        result.reserve(static_cast<std::size_t>(n_allowed(fi)));

        // Remove current field's contribution, then add each local variant
        int32_t current_local = static_cast<int32_t>(
            decode(dense_id)[fi] == Strategy{} ? 0 :
            local_maps[fi].to(decode(dense_id)[fi]));
        int32_t base_id = dense_id - current_local * strides[fi];

        for (int k = 0; k < n_allowed(fi); ++k)
            result.push_back(base_id + k * strides[fi]);

        return result;
    }
};

// ─────────────────────────────────────────────────────────────────────────────
//  Pre-built schema indices for the two concrete schemas
// ─────────────────────────────────────────────────────────────────────────────

inline constexpr SchemaIndex<4>  trivial_index{ trivial_schema };
inline constexpr SchemaIndex<12> full_index   { full_schema    };

// ─────────────────────────────────────────────────────────────────────────────
//  CostTable<N, T> — flat DP cost table indexed by dense_id
//
//  Wraps a std::vector<T> of size index.size(), providing named access
//  via encode(plan) and decode(dense_id).
//
//  T is typically:
//    double          — measured p99 latency
//    float           — SVR predicted cost
//    std::optional<double> — sparse table (unmeasured = nullopt)
// ─────────────────────────────────────────────────────────────────────────────

template<int N, typename T>
class CostTable {
public:
    explicit CostTable(const SchemaIndex<N>& idx, T fill = T{})
        : index_{&idx}
        , data_(static_cast<std::size_t>(idx.size()), fill)
    {}

    // ── Element access ───────────────────────────────────────────────────────

    [[nodiscard]] T& at(int32_t dense_id) {
        if (!index_->is_valid_id(dense_id))
            throw std::out_of_range("CostTable: dense_id out of range");
        return data_[static_cast<std::size_t>(dense_id)];
    }

    [[nodiscard]] const T& at(int32_t dense_id) const {
        if (!index_->is_valid_id(dense_id))
            throw std::out_of_range("CostTable: dense_id out of range");
        return data_[static_cast<std::size_t>(dense_id)];
    }

    [[nodiscard]] T& operator[](int32_t dense_id) noexcept {
        return data_[static_cast<std::size_t>(dense_id)];
    }

    [[nodiscard]] const T& operator[](int32_t dense_id) const noexcept {
        return data_[static_cast<std::size_t>(dense_id)];
    }

    // Plan-addressed access
    [[nodiscard]] T& at_plan(const std::array<Strategy, N>& plan) {
        int32_t id = index_->encode(plan);
        if (id < 0)
            throw std::invalid_argument("CostTable: plan contains disallowed strategy");
        return data_[static_cast<std::size_t>(id)];
    }

    [[nodiscard]] const T& at_plan(const std::array<Strategy, N>& plan) const {
        int32_t id = index_->encode(plan);
        if (id < 0)
            throw std::invalid_argument("CostTable: plan contains disallowed strategy");
        return data_[static_cast<std::size_t>(id)];
    }

    // ── Properties ───────────────────────────────────────────────────────────

    [[nodiscard]] int32_t size() const noexcept { return index_->size(); }

    [[nodiscard]] const SchemaIndex<N>& index() const noexcept { return *index_; }

    [[nodiscard]] std::span<T>       span()       noexcept { return data_; }
    [[nodiscard]] std::span<const T> span() const noexcept { return data_; }

    // Best entry: returns {dense_id, value} for min-cost entry.
    // Requires T to be less-than comparable.
    [[nodiscard]] std::pair<int32_t, T> best() const
        requires std::totally_ordered<T>
    {
        if (data_.empty())
            throw std::logic_error("CostTable::best on empty table");
        int32_t best_id = 0;
        for (int32_t i = 1; i < static_cast<int32_t>(data_.size()); ++i)
            if (data_[static_cast<std::size_t>(i)] <
                data_[static_cast<std::size_t>(best_id)])
                best_id = i;
        return {best_id, data_[static_cast<std::size_t>(best_id)]};
    }

private:
    const SchemaIndex<N>* index_;
    std::vector<T>        data_;
};

} // namespace ctdp::calibrator::fix

#endif // CTDP_CALIBRATOR_FIX_STRATEGY_IDS_H
