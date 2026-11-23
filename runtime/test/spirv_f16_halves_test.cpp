// Pure-CPU tests for runtime/src/spirv_f16_halves.h (SHADER-378):
//   - fast path when PS has no f16_halves or VS already matches all PS half locations;
//   - in-place relocation when VS already emits the requested params at different locations;
//   - 16-param VS (param0..param15, static halves only for 0..6) paired with a corpus-style PS (raw 0..14,
//     f16 on 0, 4, 5, 6, 7): unused static halves demoted to Private, reused halves relocated, missing param 7
//     synthesized with OpLoad + 4x UnpackHalf2x16 + OpStore before OpReturn;
//   - PS relocation when a PS half location collides with an unread raw VS output;
//   - SPIR-V 1.3 vs 1.4+ OpEntryPoint interface rules for demoted Private variables;
//   - clean failure on genuine >30 location budget overflow (leaving input buffers untouched).
// No GPU or game files required.
#include "../src/spirv_f16_halves.h"

#include <cstdint>
#include <cstdio>
#include <map>
#include <set>
#include <string>
#include <vector>

#define CHECK(cond) do { \
    if (!(cond)) { std::fprintf(stderr, "line %d: %s\n", __LINE__, #cond); return 1; } \
} while (0)

namespace {

using namespace leonida::spirv_detail;

void append_inst(std::vector<std::uint32_t>& w, std::uint16_t op, std::initializer_list<std::uint32_t> args) {
    w.push_back((std::uint32_t(1 + args.size()) << 16) | op);
    for (std::uint32_t a : args) w.push_back(a);
}

void append_inst_vec(std::vector<std::uint32_t>& w, std::uint16_t op, const std::vector<std::uint32_t>& args) {
    w.push_back((std::uint32_t(1 + args.size()) << 16) | op);
    w.insert(w.end(), args.begin(), args.end());
}

// Build a minimal valid Vertex SPIR-V module with `num_raw_params` Output vec4 params (Location 0..num_raw_params-1)
// and optional static f16_halves Output vec4 vars at the given locations.
std::vector<std::uint32_t> build_vs(std::uint32_t version, std::uint32_t num_raw_params,
                                    const std::map<std::uint32_t, std::array<std::uint32_t, 2>>& static_halves) {
    // IDs:
    // 1 = %void, 2 = %fn, 3 = %float, 4 = %v4float, 5 = %ptr_out_v4, 6 = %null_v4, 7 = %main, 8 = %entry
    // 9 = %out_pos (BuiltIn Position)
    // 10.. = raw param vars, then static half vars (lo, hi)
    std::uint32_t next_id = 10;
    std::vector<std::uint32_t> raw_vars;
    for (std::uint32_t p = 0; p < num_raw_params; ++p) raw_vars.push_back(next_id++);
    std::vector<std::pair<std::uint32_t, std::uint32_t>> half_vars;  // (var_id, loc)
    for (const auto& [param, pair] : static_halves) {
        half_vars.push_back({next_id++, pair[0]});
        half_vars.push_back({next_id++, pair[1]});
    }

    std::vector<std::uint32_t> w = {kSpirvMagic, version, 0, next_id, 0};
    append_inst(w, kOpCapability, {1});            // Shader
    append_inst(w, kOpMemoryModel, {0, 1});        // Logical GLSL450
    std::vector<std::uint32_t> ep = {kExecModelVertex, 7, 0x6e69616du, 0x00000000u, 9};  // "main\0"
    for (std::uint32_t v : raw_vars) ep.push_back(v);
    for (const auto& [v, loc] : half_vars) ep.push_back(v);
    append_inst_vec(w, kOpEntryPoint, ep);

    append_inst(w, kOpDecorate, {9, 11, 0});  // BuiltIn Position
    for (std::uint32_t p = 0; p < num_raw_params; ++p) append_inst(w, kOpDecorate, {raw_vars[p], kDecorationLocation, p});
    for (const auto& [v, loc] : half_vars) append_inst(w, kOpDecorate, {v, kDecorationLocation, loc});

    append_inst(w, 19, {1});                       // OpTypeVoid
    append_inst(w, 33, {2, 1});                    // OpTypeFunction %void
    append_inst(w, kOpTypeFloat, {3, 32});
    append_inst(w, kOpTypeVector, {4, 3, 4});
    append_inst(w, kOpTypePointer, {5, kStorageOutput, 4});
    append_inst(w, kOpConstantNull, {4, 6});
    append_inst(w, kOpVariable, {5, 9, kStorageOutput});
    for (std::uint32_t v : raw_vars) append_inst(w, kOpVariable, {5, v, kStorageOutput});
    for (const auto& [v, loc] : half_vars) append_inst(w, kOpVariable, {5, v, kStorageOutput});

    append_inst(w, kOpFunction, {1, 7, 0, 2});
    append_inst(w, 248, {8});                      // OpLabel
    append_inst(w, kOpStore, {9, 6});
    for (std::uint32_t v : raw_vars) append_inst(w, kOpStore, {v, 6});
    for (const auto& [v, loc] : half_vars) append_inst(w, kOpStore, {v, 6});
    append_inst(w, kOpReturn, {});
    append_inst(w, kOpFunctionEnd, {});
    return w;
}

// Build a minimal valid Fragment SPIR-V module with `raw_in_locs` and `half_in_locs` Input vec4 variables.
std::vector<std::uint32_t> build_ps(std::uint32_t version, const std::vector<std::uint32_t>& raw_in_locs,
                                    const std::map<std::uint32_t, std::array<std::uint32_t, 2>>& f16_halves) {
    // 1 = %void, 2 = %fn, 3 = %float, 4 = %v4float, 5 = %ptr_in_v4, 6 = %ptr_out_v4, 7 = %null_v4, 8 = %out0,
    // 9 = %main, 10 = %entry
    std::uint32_t next_id = 11;
    std::vector<std::pair<std::uint32_t, std::uint32_t>> in_vars;  // (var_id, loc)
    for (std::uint32_t loc : raw_in_locs) in_vars.push_back({next_id++, loc});
    for (const auto& [param, pair] : f16_halves) {
        in_vars.push_back({next_id++, pair[0]});
        in_vars.push_back({next_id++, pair[1]});
    }

    std::vector<std::uint32_t> w = {kSpirvMagic, version, 0, next_id, 0};
    append_inst(w, kOpCapability, {1});
    append_inst(w, kOpMemoryModel, {0, 1});
    std::vector<std::uint32_t> ep = {kExecModelFragment, 9, 0x6e69616du, 0x00000000u, 8};
    for (const auto& [v, loc] : in_vars) ep.push_back(v);
    append_inst_vec(w, kOpEntryPoint, ep);
    append_inst(w, kOpExecutionMode, {9, 7});  // OriginUpperLeft

    append_inst(w, kOpDecorate, {8, kDecorationLocation, 0});
    for (const auto& [v, loc] : in_vars) append_inst(w, kOpDecorate, {v, kDecorationLocation, loc});

    append_inst(w, 19, {1});
    append_inst(w, 33, {2, 1});
    append_inst(w, kOpTypeFloat, {3, 32});
    append_inst(w, kOpTypeVector, {4, 3, 4});
    append_inst(w, kOpTypePointer, {5, kStorageInput, 4});
    append_inst(w, kOpTypePointer, {6, kStorageOutput, 4});
    append_inst(w, kOpConstantNull, {4, 7});
    append_inst(w, kOpVariable, {6, 8, kStorageOutput});
    for (const auto& [v, loc] : in_vars) append_inst(w, kOpVariable, {5, v, kStorageInput});

    append_inst(w, kOpFunction, {1, 9, 0, 2});
    append_inst(w, 248, {10});
    append_inst(w, kOpStore, {8, 7});
    append_inst(w, kOpReturn, {});
    append_inst(w, kOpFunctionEnd, {});
    return w;
}

bool is_interface_dec_for_test(std::uint32_t dec) {
    switch (dec) {
    case 11:    // BuiltIn
    case 13:    // NoPerspective
    case 14:    // Flat
    case 15:    // Patch
    case 16:    // Centroid
    case 17:    // Sample
    case 18:    // Invariant
    case 29:    // Stream
    case 30:    // Location
    case 31:    // Component
    case 32:    // Index
    case 36:    // XfbBuffer
    case 37:    // XfbStride
    case 5271:  // PerPrimitiveEXT
    case 5285:  // PerVertexKHR
        return true;
    default:
        return false;
    }
}

// Verify basic SPIR-V structural invariants: valid section ordering, all IDs < bound, unique Locations per storage,
// and no interface-only decorations remaining on Private variables (directly or via OpGroupDecorate).
bool verify_spirv_structure(const std::vector<std::uint32_t>& words, std::uint32_t storage_class,
                            std::map<std::uint32_t, std::uint32_t>& out_loc_to_var, int& out_unpack_count,
                            int& out_private_vars, std::set<std::uint32_t>& out_ep_iface) {
    Module mod;
    if (!mod.parse(words)) return false;
    out_loc_to_var.clear();
    out_unpack_count = 0;
    out_private_vars = 0;
    out_ep_iface.clear();

    // Check section order: pre-type -> types/globals -> functions, and every defined type appears before its use.
    int section = 0;  // 0 = pre-type, 1 = types/globals, 2 = functions
    std::set<std::uint32_t> defined_types;
    std::map<std::uint32_t, std::uint32_t> var_storage;
    std::set<std::uint32_t> dec_groups;
    std::set<std::uint32_t> iface_dec_groups;
    std::uint32_t glsl_id = 0;
    for (const Inst& inst : mod.insts) {
        if (inst.opcode == kOpExtInstImport && inst.words.size() >= 3 &&
            decode_literal_string(std::span<const std::uint32_t>(inst.words).subspan(2)) == "GLSL.std.450") {
            glsl_id = inst.words[1];
        }
        if (inst.opcode == kOpEntryPoint) {
            const std::size_t s = entry_point_interface_start(inst);
            for (std::size_t i = s; i < inst.words.size(); ++i) out_ep_iface.insert(inst.words[i]);
        }
        if (inst.opcode == kOpDecorationGroup && inst.words.size() >= 2) dec_groups.insert(inst.words[1]);
        if (inst.opcode == kOpDecorate && inst.words.size() >= 3 && is_interface_dec_for_test(inst.words[2])) {
            iface_dec_groups.insert(inst.words[1]);
        }
        if (inst.opcode == kOpFunction) {
            section = 2;
        } else if (section == 0 && !is_pre_type_opcode(inst.opcode)) {
            section = 1;
        } else if (section == 1 && is_pre_type_opcode(inst.opcode)) {
            return false;  // annotation/header appeared after types started
        }
        if (section == 1) {
            if (inst.opcode == 19 || inst.opcode == 33 || inst.opcode == 20 || inst.opcode == kOpTypeInt ||
                inst.opcode == kOpTypeFloat || inst.opcode == kOpTypeVector || inst.opcode == 24 ||
                inst.opcode == 28 || inst.opcode == 30 || inst.opcode == kOpTypePointer) {
                defined_types.insert(inst.words[1]);
                if (inst.words[1] >= mod.header[3]) return false;
            }
            if (inst.opcode == kOpVariable) {
                if (!defined_types.count(inst.words[1])) return false;  // pointer type must precede OpVariable
                if (inst.words[2] >= mod.header[3]) return false;
                var_storage[inst.words[2]] = inst.words[3];
                if (inst.words[3] == kStoragePrivate) ++out_private_vars;
            }
        }
        if (inst.opcode == kOpExtInst && inst.words.size() >= 6 && inst.words[3] == glsl_id &&
            inst.words[4] == kGlslUnpackHalf2x16) {
            ++out_unpack_count;
            if (inst.words[2] >= mod.header[3]) return false;
        }
    }
    for (const Inst& inst : mod.insts) {
        if (inst.opcode == kOpDecorate && inst.words.size() >= 3) {
            const auto it = var_storage.find(inst.words[1]);
            if (it != var_storage.end() && it->second == storage_class && inst.words[2] == kDecorationLocation &&
                inst.words.size() >= 4) {
                if (!out_loc_to_var.emplace(inst.words[3], inst.words[1]).second) return false;  // duplicate Location
            }
            if (it != var_storage.end() && it->second == kStoragePrivate &&
                is_interface_dec_for_test(inst.words[2])) {
                return false;  // Private cannot carry interface-only decorations
            }
        } else if (inst.opcode == kOpGroupDecorate && inst.words.size() >= 2 &&
                   dec_groups.count(inst.words[1]) && iface_dec_groups.count(inst.words[1])) {
            for (std::size_t i = 2; i < inst.words.size(); ++i) {
                const auto it = var_storage.find(inst.words[i]);
                if (it != var_storage.end() && it->second == kStoragePrivate) return false;
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    using namespace leonida;

    // 1. Fast path: already matching static f16_halves -> both modules untouched.
    {
        ShaderEntry vs_e, ps_e;
        vs_e.f16_halves = {{0, {28, 29}}, {1, {26, 27}}};
        ps_e.f16_halves = {{0, {28, 29}}};
        auto vs_w = build_vs(0x00010500u, 2, vs_e.f16_halves);
        auto ps_w = build_ps(0x00010500u, {0, 1}, ps_e.f16_halves);
        const auto vs_orig = vs_w, ps_orig = ps_w;
        std::string err;
        CHECK(link_f16_halves(vs_w, vs_e, ps_w, ps_e, &err));
        CHECK(vs_w == vs_orig && ps_w == ps_orig);
    }

    // 2. Corpus pattern (SHADER-378): 16-param VS (param0..param15, static halves only for 0..6) paired with a PS
    //    reading raw attrs 0..14 and f16 attrs {0, 4, 5, 6, 7} packed at 20..29.
    //    Expected: params 1,2,3 demoted to Private (6 Private vars), params 0,4,5,6 relocated, param 7 synthesized
    //    with 4 UnpackHalf2x16 ops at Locations 20 & 21; PS unchanged.
    {
        ShaderEntry vs_e, ps_e;
        for (std::uint32_t n = 0; n <= 6; ++n) vs_e.f16_halves[n] = {28 - 2 * n, 29 - 2 * n};
        ps_e.f16_halves = {{0, {28, 29}}, {4, {26, 27}}, {5, {24, 25}}, {6, {22, 23}}, {7, {20, 21}}};
        std::vector<std::uint32_t> ps_raw;
        for (std::uint32_t a = 0; a < 15; ++a) ps_raw.push_back(a);

        auto vs_w = build_vs(0x00010500u, 16, vs_e.f16_halves);
        auto ps_w = build_ps(0x00010500u, ps_raw, ps_e.f16_halves);
        const auto ps_orig = ps_w;
        std::string err;
        CHECK(link_f16_halves(vs_w, vs_e, ps_w, ps_e, &err));
        CHECK(ps_w == ps_orig);  // PS locations 20..29 do not collide with VS raw 0..15

        std::map<std::uint32_t, std::uint32_t> vs_locs;
        int unpacks = 0, privs = 0;
        std::set<std::uint32_t> ep_iface;
        CHECK(verify_spirv_structure(vs_w, kStorageOutput, vs_locs, unpacks, privs, ep_iface));
        CHECK(privs == 6);    // 3 unneeded static half pairs (params 1, 2, 3) demoted to Private
        CHECK(unpacks == 4);  // 1 missing param (param 7) * 4 vec4 channels
        for (std::uint32_t loc = 0; loc < 16; ++loc) CHECK(vs_locs.count(loc) == 1);
        for (std::uint32_t loc = 20; loc <= 29; ++loc) {
            CHECK(vs_locs.count(loc) == 1);
            CHECK(ep_iface.count(vs_locs[loc]) == 1);
        }
        CHECK(vs_locs.size() == 26);  // 16 raw + 10 decoded halves; all <= 29
        CHECK(vs_locs.rbegin()->first == 29);
    }

    // 3. PS relocation: VS exports raw param0..param15, PS only read raw attr0..13 so PS placed attr7 at [14, 15]
    //    (colliding with VS's raw param14/param15). Both VS and PS must relocate attr7 to [28, 29].
    {
        ShaderEntry vs_e, ps_e;
        ps_e.f16_halves = {{7, {14, 15}}};
        std::vector<std::uint32_t> ps_raw;
        for (std::uint32_t a = 0; a < 14; ++a) ps_raw.push_back(a);

        auto vs_w = build_vs(0x00010500u, 16, {});
        auto ps_w = build_ps(0x00010500u, ps_raw, ps_e.f16_halves);
        std::string err;
        CHECK(link_f16_halves(vs_w, vs_e, ps_w, ps_e, &err));

        std::map<std::uint32_t, std::uint32_t> vs_locs, ps_locs;
        int vs_unpacks = 0, vs_privs = 0, ps_unpacks = 0, ps_privs = 0;
        std::set<std::uint32_t> vs_ep, ps_ep;
        CHECK(verify_spirv_structure(vs_w, kStorageOutput, vs_locs, vs_unpacks, vs_privs, vs_ep));
        CHECK(verify_spirv_structure(ps_w, kStorageInput, ps_locs, ps_unpacks, ps_privs, ps_ep));
        CHECK(vs_unpacks == 4);
        CHECK(vs_locs.count(28) == 1 && vs_locs.count(29) == 1);
        CHECK(ps_locs.count(28) == 1 && ps_locs.count(29) == 1);
        CHECK(!ps_locs.count(14) && !ps_locs.count(15));
    }

    // 4. SPIR-V 1.3 OpEntryPoint drops demoted Private variables from interface list (SPIR-V 1.4+ keeps them).
    {
        ShaderEntry vs_e, ps_e;
        vs_e.f16_halves = {{0, {28, 29}}, {1, {26, 27}}};
        ps_e.f16_halves = {{0, {26, 27}}};  // force patching; param 1 is demoted
        auto vs_13 = build_vs(0x00010300u, 2, vs_e.f16_halves);
        auto ps_13 = build_ps(0x00010300u, {0, 1}, ps_e.f16_halves);
        std::string err;
        CHECK(link_f16_halves(vs_13, vs_e, ps_13, ps_e, &err));
        std::map<std::uint32_t, std::uint32_t> locs;
        int unpacks = 0, privs = 0;
        std::set<std::uint32_t> ep_iface;
        CHECK(verify_spirv_structure(vs_13, kStorageOutput, locs, unpacks, privs, ep_iface));
        CHECK(privs == 2);
        // Position (9) + 2 raw (10,11) + 2 reused halves (12,13) = 5 interface vars; demoted 14,15 dropped in v1.3.
        CHECK(ep_iface.size() == 5);
    }

    // 5. Genuine budget overflow (>30 locations needed): returns false, leaves modules unchanged.
    {
        ShaderEntry vs_e, ps_e;
        ps_e.f16_halves = {{0, {28, 29}}};
        auto vs_w = build_vs(0x00010500u, 29, {});  // 29 raw outputs (0..28) -> only location 29 is free (< 2 needed)
        auto ps_w = build_ps(0x00010500u, {0}, ps_e.f16_halves);
        const auto vs_orig = vs_w, ps_orig = ps_w;
        std::string err;
        CHECK(!link_f16_halves(vs_w, vs_e, ps_w, ps_e, &err));
        CHECK(!err.empty());
        CHECK(vs_w == vs_orig && ps_w == ps_orig);
    }

    // 6. Fast-path budget enforcement: matched VS/PS half pair > top_location (30, 31) must be relocated into 0..29.
    {
        ShaderEntry vs_e, ps_e;
        vs_e.f16_halves = {{0, {30, 31}}};
        ps_e.f16_halves = {{0, {30, 31}}};
        auto vs_w = build_vs(0x00010500u, 2, vs_e.f16_halves);
        auto ps_w = build_ps(0x00010500u, {0, 1}, ps_e.f16_halves);
        std::string err;
        CHECK(link_f16_halves(vs_w, vs_e, ps_w, ps_e, &err));

        std::map<std::uint32_t, std::uint32_t> vs_locs, ps_locs;
        int vs_u = 0, vs_p = 0, ps_u = 0, ps_p = 0;
        std::set<std::uint32_t> vs_ep, ps_ep;
        CHECK(verify_spirv_structure(vs_w, kStorageOutput, vs_locs, vs_u, vs_p, vs_ep));
        CHECK(verify_spirv_structure(ps_w, kStorageInput, ps_locs, ps_u, ps_p, ps_ep));
        CHECK(!vs_locs.empty() && vs_locs.rbegin()->first <= 29);
        CHECK(!ps_locs.empty() && ps_locs.rbegin()->first <= 29);
        CHECK(vs_locs.count(28) == 1 && vs_locs.count(29) == 1);
        CHECK(ps_locs.count(28) == 1 && ps_locs.count(29) == 1);
        CHECK(!vs_locs.count(30) && !vs_locs.count(31));
        CHECK(!ps_locs.count(30) && !ps_locs.count(31));
    }

    // 7. Fast-path budget enforcement: extra unused VS static half pair > top_location must be demoted to Private.
    {
        ShaderEntry vs_e, ps_e;
        vs_e.f16_halves = {{0, {28, 29}}, {1, {30, 31}}};
        ps_e.f16_halves = {{0, {28, 29}}};
        auto vs_w = build_vs(0x00010500u, 2, vs_e.f16_halves);
        auto ps_w = build_ps(0x00010500u, {0, 1}, ps_e.f16_halves);
        const auto ps_orig = ps_w;
        std::string err;
        CHECK(link_f16_halves(vs_w, vs_e, ps_w, ps_e, &err));
        CHECK(ps_w == ps_orig);

        std::map<std::uint32_t, std::uint32_t> vs_locs;
        int unpacks = 0, privs = 0;
        std::set<std::uint32_t> ep_iface;
        CHECK(verify_spirv_structure(vs_w, kStorageOutput, vs_locs, unpacks, privs, ep_iface));
        CHECK(privs == 2);
        CHECK(unpacks == 0);
        CHECK(vs_locs.count(28) == 1 && vs_locs.count(29) == 1);
        CHECK(vs_locs.rbegin()->first <= 29);
    }

    // 8. Empty ps_entry.f16_halves: fast-path untouched when VS halves fit <= top_location; demote when > top_location.
    {
        ShaderEntry vs_e, ps_e;
        vs_e.f16_halves = {{0, {28, 29}}};
        auto vs_w = build_vs(0x00010500u, 2, vs_e.f16_halves);
        auto ps_w = build_ps(0x00010500u, {0, 1}, {});
        const auto vs_orig = vs_w, ps_orig = ps_w;
        std::string err;
        CHECK(link_f16_halves(vs_w, vs_e, ps_w, ps_e, &err));
        CHECK(vs_w == vs_orig && ps_w == ps_orig);

        ShaderEntry vs_over_e;
        vs_over_e.f16_halves = {{0, {30, 31}}};
        auto vs_over = build_vs(0x00010500u, 2, vs_over_e.f16_halves);
        auto ps_over = build_ps(0x00010500u, {0, 1}, {});
        const auto ps_over_orig = ps_over;
        CHECK(link_f16_halves(vs_over, vs_over_e, ps_over, ps_e, &err));
        CHECK(ps_over == ps_over_orig);

        std::map<std::uint32_t, std::uint32_t> vs_locs;
        int unpacks = 0, privs = 0;
        std::set<std::uint32_t> ep_iface;
        CHECK(verify_spirv_structure(vs_over, kStorageOutput, vs_locs, unpacks, privs, ep_iface));
        CHECK(privs == 2);
        CHECK(vs_locs.rbegin()->first <= 29);
    }

    // 9. Demoted Private variables strip all interface-only OpDecorate & OpGroupDecorate (and orphaned groups),
    //    while preserving non-interface decorations such as RelaxedPrecision and shared group targets.
    {
        // IDs: 1=%void 2=%fn 3=%float 4=%v4float 5=%ptr_out_v4 6=%null_v4 7=%main 8=%entry 9=%pos
        //      10=%raw0 11=%raw1 12=%h0_lo 13=%h0_hi 14=%h1_lo 15=%h1_hi 16=%grp_shared 17=%grp_excl
        std::vector<std::uint32_t> vs_w = {kSpirvMagic, 0x00010500u, 0, 18, 0};
        append_inst(vs_w, kOpCapability, {1});
        append_inst(vs_w, kOpMemoryModel, {0, 1});
        append_inst(vs_w, kOpEntryPoint, {kExecModelVertex, 7, 0x6e69616du, 0, 9, 10, 11, 12, 13, 14, 15});
        append_inst(vs_w, kOpDecorate, {9, 11, 0});  // BuiltIn Position
        append_inst(vs_w, kOpDecorate, {10, kDecorationLocation, 0});
        append_inst(vs_w, kOpDecorate, {11, kDecorationLocation, 1});
        append_inst(vs_w, kOpDecorate, {12, kDecorationLocation, 28});
        append_inst(vs_w, kOpDecorate, {13, kDecorationLocation, 29});
        append_inst(vs_w, kOpDecorate, {14, kDecorationLocation, 26});
        append_inst(vs_w, kOpDecorate, {14, 14});     // Flat
        append_inst(vs_w, kOpDecorate, {14, 31, 0});  // Component 0
        append_inst(vs_w, kOpDecorate, {14, 18});     // Invariant
        append_inst(vs_w, kOpDecorate, {14, 0});      // RelaxedPrecision (must survive)
        append_inst(vs_w, kOpDecorate, {15, kDecorationLocation, 27});
        append_inst(vs_w, kOpDecorate, {16, 13});     // NoPerspective on grp_shared
        append_inst(vs_w, kOpDecorationGroup, {16});
        append_inst(vs_w, kOpGroupDecorate, {16, 10, 15});  // raw0 (stays Output) + h1_hi (demoted)
        append_inst(vs_w, kOpDecorate, {17, 18});           // Invariant on grp_excl
        append_inst(vs_w, kOpDecorationGroup, {17});
        append_inst(vs_w, kOpGroupDecorate, {17, 14, 15});  // only demoted h1_lo + h1_hi -> orphaned
        append_inst(vs_w, 19, {1});
        append_inst(vs_w, 33, {2, 1});
        append_inst(vs_w, kOpTypeFloat, {3, 32});
        append_inst(vs_w, kOpTypeVector, {4, 3, 4});
        append_inst(vs_w, kOpTypePointer, {5, kStorageOutput, 4});
        append_inst(vs_w, kOpConstantNull, {4, 6});
        for (std::uint32_t v : {9u, 10u, 11u, 12u, 13u, 14u, 15u}) append_inst(vs_w, kOpVariable, {5, v, kStorageOutput});
        append_inst(vs_w, kOpFunction, {1, 7, 0, 2});
        append_inst(vs_w, 248, {8});
        for (std::uint32_t v : {9u, 10u, 11u, 12u, 13u, 14u, 15u}) append_inst(vs_w, kOpStore, {v, 6});
        append_inst(vs_w, kOpReturn, {});
        append_inst(vs_w, kOpFunctionEnd, {});

        ShaderEntry vs_e, ps_e;
        vs_e.f16_halves = {{0, {28, 29}}, {1, {26, 27}}};
        ps_e.f16_halves = {{0, {26, 27}}};  // demotes param 1 (14, 15)
        auto ps_w = build_ps(0x00010500u, {0, 1}, ps_e.f16_halves);
        std::string err;
        CHECK(link_f16_halves(vs_w, vs_e, ps_w, ps_e, &err));

        std::map<std::uint32_t, std::uint32_t> vs_locs;
        int unpacks = 0, privs = 0;
        std::set<std::uint32_t> ep_iface;
        CHECK(verify_spirv_structure(vs_w, kStorageOutput, vs_locs, unpacks, privs, ep_iface));
        CHECK(privs == 2);

        Module out_mod;
        CHECK(out_mod.parse(vs_w));
        bool has_relaxed_on_14 = false;
        bool has_shared_grp = false;
        bool shared_keeps_10 = false;
        bool shared_drops_15 = true;
        bool has_excl_grp = false;
        for (const Inst& inst : out_mod.insts) {
            if (inst.opcode == kOpDecorate && inst.words.size() >= 3 && inst.words[1] == 14 && inst.words[2] == 0)
                has_relaxed_on_14 = true;
            if (inst.opcode == kOpDecorationGroup && inst.words.size() >= 2 && inst.words[1] == 16)
                has_shared_grp = true;
            if (inst.opcode == kOpDecorationGroup && inst.words.size() >= 2 && inst.words[1] == 17)
                has_excl_grp = true;
            if (inst.opcode == kOpDecorate && inst.words.size() >= 2 && inst.words[1] == 17) has_excl_grp = true;
            if (inst.opcode == kOpGroupDecorate && inst.words.size() >= 2 && inst.words[1] == 17) has_excl_grp = true;
            if (inst.opcode == kOpGroupDecorate && inst.words.size() >= 2 && inst.words[1] == 16) {
                for (std::size_t i = 2; i < inst.words.size(); ++i) {
                    if (inst.words[i] == 10) shared_keeps_10 = true;
                    if (inst.words[i] == 15) shared_drops_15 = false;
                }
            }
        }
        CHECK(has_relaxed_on_14);
        CHECK(has_shared_grp && shared_keeps_10 && shared_drops_15);
        CHECK(!has_excl_grp);
    }

    // 10. Non-load/store pointer uses refuse demotion cleanly; literal operands do not false-positive.
    {
        ShaderEntry vs_e, ps_e;
        vs_e.f16_halves = {{0, {28, 29}}, {1, {26, 27}}};
        ps_e.f16_halves = {{0, {26, 27}}};

        // 10a: OpCopyObject alias of demoted var 14.
        {
            std::vector<std::uint32_t> vs_w = {kSpirvMagic, 0x00010500u, 0, 17, 0};
            append_inst(vs_w, kOpCapability, {1});
            append_inst(vs_w, kOpMemoryModel, {0, 1});
            append_inst(vs_w, kOpEntryPoint, {kExecModelVertex, 7, 0x6e69616du, 0, 9, 10, 11, 12, 13, 14, 15});
            append_inst(vs_w, kOpDecorate, {9, 11, 0});
            append_inst(vs_w, kOpDecorate, {10, kDecorationLocation, 0});
            append_inst(vs_w, kOpDecorate, {11, kDecorationLocation, 1});
            append_inst(vs_w, kOpDecorate, {12, kDecorationLocation, 28});
            append_inst(vs_w, kOpDecorate, {13, kDecorationLocation, 29});
            append_inst(vs_w, kOpDecorate, {14, kDecorationLocation, 26});
            append_inst(vs_w, kOpDecorate, {15, kDecorationLocation, 27});
            append_inst(vs_w, 19, {1});
            append_inst(vs_w, 33, {2, 1});
            append_inst(vs_w, kOpTypeFloat, {3, 32});
            append_inst(vs_w, kOpTypeVector, {4, 3, 4});
            append_inst(vs_w, kOpTypePointer, {5, kStorageOutput, 4});
            append_inst(vs_w, kOpConstantNull, {4, 6});
            for (std::uint32_t v : {9u, 10u, 11u, 12u, 13u, 14u, 15u}) append_inst(vs_w, kOpVariable, {5, v, kStorageOutput});
            append_inst(vs_w, kOpFunction, {1, 7, 0, 2});
            append_inst(vs_w, 248, {8});
            for (std::uint32_t v : {9u, 10u, 11u, 12u, 13u, 15u}) append_inst(vs_w, kOpStore, {v, 6});
            append_inst(vs_w, 83, {5, 16, 14});  // OpCopyObject %alias = 14
            append_inst(vs_w, kOpStore, {16, 6});
            append_inst(vs_w, kOpReturn, {});
            append_inst(vs_w, kOpFunctionEnd, {});

            auto ps_w = build_ps(0x00010500u, {0, 1}, ps_e.f16_halves);
            const auto vs_orig = vs_w, ps_orig = ps_w;
            std::string err;
            CHECK(!link_f16_halves(vs_w, vs_e, ps_w, ps_e, &err));
            CHECK(!err.empty());
            CHECK(vs_w == vs_orig && ps_w == ps_orig);
        }

        // 10b: OpFunctionCall passing demoted var 14 to a helper function.
        {
            // 18=%helper_fn_ty, 19=%helper, 20=%param, 21=%helper_entry
            std::vector<std::uint32_t> vs_w = {kSpirvMagic, 0x00010500u, 0, 22, 0};
            append_inst(vs_w, kOpCapability, {1});
            append_inst(vs_w, kOpMemoryModel, {0, 1});
            append_inst(vs_w, kOpEntryPoint, {kExecModelVertex, 7, 0x6e69616du, 0, 9, 10, 11, 12, 13, 14, 15});
            append_inst(vs_w, kOpDecorate, {9, 11, 0});
            append_inst(vs_w, kOpDecorate, {10, kDecorationLocation, 0});
            append_inst(vs_w, kOpDecorate, {11, kDecorationLocation, 1});
            append_inst(vs_w, kOpDecorate, {12, kDecorationLocation, 28});
            append_inst(vs_w, kOpDecorate, {13, kDecorationLocation, 29});
            append_inst(vs_w, kOpDecorate, {14, kDecorationLocation, 26});
            append_inst(vs_w, kOpDecorate, {15, kDecorationLocation, 27});
            append_inst(vs_w, 19, {1});
            append_inst(vs_w, 33, {2, 1});
            append_inst(vs_w, kOpTypeFloat, {3, 32});
            append_inst(vs_w, kOpTypeVector, {4, 3, 4});
            append_inst(vs_w, kOpTypePointer, {5, kStorageOutput, 4});
            append_inst(vs_w, 33, {18, 1, 5});  // void(ptr_out_v4)
            append_inst(vs_w, kOpConstantNull, {4, 6});
            for (std::uint32_t v : {9u, 10u, 11u, 12u, 13u, 14u, 15u}) append_inst(vs_w, kOpVariable, {5, v, kStorageOutput});
            append_inst(vs_w, kOpFunction, {1, 7, 0, 2});
            append_inst(vs_w, 248, {8});
            for (std::uint32_t v : {9u, 10u, 11u, 12u, 13u, 15u}) append_inst(vs_w, kOpStore, {v, 6});
            append_inst(vs_w, 57, {1, 17, 19, 14});  // OpFunctionCall %helper(14)
            append_inst(vs_w, kOpReturn, {});
            append_inst(vs_w, kOpFunctionEnd, {});
            append_inst(vs_w, kOpFunction, {1, 19, 0, 18});
            append_inst(vs_w, 55, {5, 20});  // OpFunctionParameter
            append_inst(vs_w, 248, {21});
            append_inst(vs_w, kOpStore, {20, 6});
            append_inst(vs_w, kOpReturn, {});
            append_inst(vs_w, kOpFunctionEnd, {});

            auto ps_w = build_ps(0x00010500u, {0, 1}, ps_e.f16_halves);
            const auto vs_orig = vs_w, ps_orig = ps_w;
            std::string err;
            CHECK(!link_f16_halves(vs_w, vs_e, ps_w, ps_e, &err));
            CHECK(!err.empty());
            CHECK(vs_w == vs_orig && ps_w == ps_orig);
        }

        // 10c: OpSelect picking between two Output pointers.
        {
            // 16=%bool, 17=%true, 18=%sel
            std::vector<std::uint32_t> vs_w = {kSpirvMagic, 0x00010500u, 0, 19, 0};
            append_inst(vs_w, kOpCapability, {1});
            append_inst(vs_w, kOpMemoryModel, {0, 1});
            append_inst(vs_w, kOpEntryPoint, {kExecModelVertex, 7, 0x6e69616du, 0, 9, 10, 11, 12, 13, 14, 15});
            append_inst(vs_w, kOpDecorate, {9, 11, 0});
            append_inst(vs_w, kOpDecorate, {10, kDecorationLocation, 0});
            append_inst(vs_w, kOpDecorate, {11, kDecorationLocation, 1});
            append_inst(vs_w, kOpDecorate, {12, kDecorationLocation, 28});
            append_inst(vs_w, kOpDecorate, {13, kDecorationLocation, 29});
            append_inst(vs_w, kOpDecorate, {14, kDecorationLocation, 26});
            append_inst(vs_w, kOpDecorate, {15, kDecorationLocation, 27});
            append_inst(vs_w, 19, {1});
            append_inst(vs_w, 20, {16});  // OpTypeBool
            append_inst(vs_w, 33, {2, 1});
            append_inst(vs_w, kOpTypeFloat, {3, 32});
            append_inst(vs_w, kOpTypeVector, {4, 3, 4});
            append_inst(vs_w, kOpTypePointer, {5, kStorageOutput, 4});
            append_inst(vs_w, 41, {16, 17});  // OpConstantTrue
            append_inst(vs_w, kOpConstantNull, {4, 6});
            for (std::uint32_t v : {9u, 10u, 11u, 12u, 13u, 14u, 15u}) append_inst(vs_w, kOpVariable, {5, v, kStorageOutput});
            append_inst(vs_w, kOpFunction, {1, 7, 0, 2});
            append_inst(vs_w, 248, {8});
            for (std::uint32_t v : {9u, 10u, 11u, 12u, 13u, 15u}) append_inst(vs_w, kOpStore, {v, 6});
            append_inst(vs_w, 169, {5, 18, 17, 14, 15});  // OpSelect %sel = %true ? 14 : 15
            append_inst(vs_w, kOpStore, {18, 6});
            append_inst(vs_w, kOpReturn, {});
            append_inst(vs_w, kOpFunctionEnd, {});

            auto ps_w = build_ps(0x00010500u, {0, 1}, ps_e.f16_halves);
            const auto vs_orig = vs_w, ps_orig = ps_w;
            std::string err;
            CHECK(!link_f16_halves(vs_w, vs_e, ps_w, ps_e, &err));
            CHECK(!err.empty());
            CHECK(vs_w == vs_orig && ps_w == ps_orig);
        }

        // 10d: Valid OpLoad + OpStore on 14 PLUS OpCompositeExtract/OpCompositeInsert whose literal index
        //      numerically equals 14 -> must NOT false-positive, and 14/15 demote cleanly.
        {
            // 16=%uint, 17=%c32, 18=%arr32_float, 19=%null_arr32, 20=%arr_ins, 21=%ext_f, 22=%ld14
            std::vector<std::uint32_t> vs_w = {kSpirvMagic, 0x00010500u, 0, 23, 0};
            append_inst(vs_w, kOpCapability, {1});
            append_inst(vs_w, kOpMemoryModel, {0, 1});
            append_inst(vs_w, kOpEntryPoint, {kExecModelVertex, 7, 0x6e69616du, 0, 9, 10, 11, 12, 13, 14, 15});
            append_inst(vs_w, kOpDecorate, {9, 11, 0});
            append_inst(vs_w, kOpDecorate, {10, kDecorationLocation, 0});
            append_inst(vs_w, kOpDecorate, {11, kDecorationLocation, 1});
            append_inst(vs_w, kOpDecorate, {12, kDecorationLocation, 28});
            append_inst(vs_w, kOpDecorate, {13, kDecorationLocation, 29});
            append_inst(vs_w, kOpDecorate, {14, kDecorationLocation, 26});
            append_inst(vs_w, kOpDecorate, {15, kDecorationLocation, 27});
            append_inst(vs_w, 19, {1});
            append_inst(vs_w, 33, {2, 1});
            append_inst(vs_w, kOpTypeInt, {16, 32, 0});
            append_inst(vs_w, kOpTypeFloat, {3, 32});
            append_inst(vs_w, kOpTypeVector, {4, 3, 4});
            append_inst(vs_w, 43, {16, 17, 32});   // OpConstant %uint 32
            append_inst(vs_w, 28, {18, 3, 17});    // OpTypeArray float[32]
            append_inst(vs_w, kOpTypePointer, {5, kStorageOutput, 4});
            append_inst(vs_w, kOpConstantNull, {4, 6});
            append_inst(vs_w, kOpConstantNull, {18, 19});
            for (std::uint32_t v : {9u, 10u, 11u, 12u, 13u, 14u, 15u}) append_inst(vs_w, kOpVariable, {5, v, kStorageOutput});
            append_inst(vs_w, kOpFunction, {1, 7, 0, 2});
            append_inst(vs_w, 248, {8});
            for (std::uint32_t v : {9u, 10u, 11u, 12u, 13u, 14u, 15u}) append_inst(vs_w, kOpStore, {v, 6});
            append_inst(vs_w, kOpLoad, {4, 22, 14});
            append_inst(vs_w, kOpCompositeExtract, {3, 21, 19, 14});  // literal index == 14
            append_inst(vs_w, 82, {18, 20, 21, 19, 14});              // OpCompositeInsert literal index == 14
            append_inst(vs_w, kOpStore, {14, 22});
            append_inst(vs_w, kOpReturn, {});
            append_inst(vs_w, kOpFunctionEnd, {});

            auto ps_w = build_ps(0x00010500u, {0, 1}, ps_e.f16_halves);
            std::string err;
            CHECK(link_f16_halves(vs_w, vs_e, ps_w, ps_e, &err));
            std::map<std::uint32_t, std::uint32_t> vs_locs;
            int unpacks = 0, privs = 0;
            std::set<std::uint32_t> ep_iface;
            CHECK(verify_spirv_structure(vs_w, kStorageOutput, vs_locs, unpacks, privs, ep_iface));
            CHECK(privs == 2);
        }
    }

    // 11. Multi-location raw interface variables (OpTypeArray on VS at 12..15, OpTypeMatrix on PS at 26..29):
    //     halves placed at {24, 25} so neither VS's vec4[4] (12..15) nor PS's mat4 (26..29) collides.
    {
        // VS: raw0..11 + arr4 Output at Location 12 (occupies 12..15)
        // IDs: 1=%void 2=%fn 3=%float 4=%v4float 5=%ptr_out_v4 6=%null_v4 7=%main 8=%entry 9=%pos
        //      10..21 = raw0..11, 22=%uint, 23=%c4, 24=%arr4, 25=%ptr_out_arr4, 26=%null_arr4, 27=%arr_var
        std::vector<std::uint32_t> vs_w = {kSpirvMagic, 0x00010500u, 0, 28, 0};
        append_inst(vs_w, kOpCapability, {1});
        append_inst(vs_w, kOpMemoryModel, {0, 1});
        std::vector<std::uint32_t> ep = {kExecModelVertex, 7, 0x6e69616du, 0, 9};
        for (std::uint32_t v = 10; v <= 21; ++v) ep.push_back(v);
        ep.push_back(27);
        append_inst_vec(vs_w, kOpEntryPoint, ep);
        append_inst(vs_w, kOpDecorate, {9, 11, 0});
        for (std::uint32_t p = 0; p < 12; ++p) append_inst(vs_w, kOpDecorate, {10 + p, kDecorationLocation, p});
        append_inst(vs_w, kOpDecorate, {27, kDecorationLocation, 12});
        append_inst(vs_w, 19, {1});
        append_inst(vs_w, 33, {2, 1});
        append_inst(vs_w, kOpTypeInt, {22, 32, 0});
        append_inst(vs_w, kOpTypeFloat, {3, 32});
        append_inst(vs_w, kOpTypeVector, {4, 3, 4});
        append_inst(vs_w, 43, {22, 23, 4});
        append_inst(vs_w, 28, {24, 4, 23});  // vec4[4]
        append_inst(vs_w, kOpTypePointer, {5, kStorageOutput, 4});
        append_inst(vs_w, kOpTypePointer, {25, kStorageOutput, 24});
        append_inst(vs_w, kOpConstantNull, {4, 6});
        append_inst(vs_w, kOpConstantNull, {24, 26});
        append_inst(vs_w, kOpVariable, {5, 9, kStorageOutput});
        for (std::uint32_t v = 10; v <= 21; ++v) append_inst(vs_w, kOpVariable, {5, v, kStorageOutput});
        append_inst(vs_w, kOpVariable, {25, 27, kStorageOutput});
        append_inst(vs_w, kOpFunction, {1, 7, 0, 2});
        append_inst(vs_w, 248, {8});
        append_inst(vs_w, kOpStore, {9, 6});
        for (std::uint32_t v = 10; v <= 21; ++v) append_inst(vs_w, kOpStore, {v, 6});
        append_inst(vs_w, kOpStore, {27, 26});
        append_inst(vs_w, kOpReturn, {});
        append_inst(vs_w, kOpFunctionEnd, {});

        // PS: raw0..11 + mat4 Input at Location 26 (occupies 26..29) + f16_halves param 0 at {14, 15}
        // IDs: 1=%void 2=%fn 3=%float 4=%v4float 5=%ptr_in_v4 6=%ptr_out_v4 7=%null_v4 8=%out0 9=%main 10=%entry
        //      11..22 = raw0..11, 23=%h_lo(14), 24=%h_hi(15), 25=%mat4, 26=%ptr_in_mat4, 27=%mat_var(26)
        std::vector<std::uint32_t> ps_w = {kSpirvMagic, 0x00010500u, 0, 28, 0};
        append_inst(ps_w, kOpCapability, {1});
        append_inst(ps_w, kOpMemoryModel, {0, 1});
        std::vector<std::uint32_t> ps_ep = {kExecModelFragment, 9, 0x6e69616du, 0, 8};
        for (std::uint32_t v = 11; v <= 24; ++v) ps_ep.push_back(v);
        ps_ep.push_back(27);
        append_inst_vec(ps_w, kOpEntryPoint, ps_ep);
        append_inst(ps_w, kOpExecutionMode, {9, 7});
        append_inst(ps_w, kOpDecorate, {8, kDecorationLocation, 0});
        for (std::uint32_t p = 0; p < 12; ++p) append_inst(ps_w, kOpDecorate, {11 + p, kDecorationLocation, p});
        append_inst(ps_w, kOpDecorate, {23, kDecorationLocation, 14});
        append_inst(ps_w, kOpDecorate, {24, kDecorationLocation, 15});
        append_inst(ps_w, kOpDecorate, {27, kDecorationLocation, 26});
        append_inst(ps_w, 19, {1});
        append_inst(ps_w, 33, {2, 1});
        append_inst(ps_w, kOpTypeFloat, {3, 32});
        append_inst(ps_w, kOpTypeVector, {4, 3, 4});
        append_inst(ps_w, 24, {25, 4, 4});  // mat4 (occupies 4 locations)
        append_inst(ps_w, kOpTypePointer, {5, kStorageInput, 4});
        append_inst(ps_w, kOpTypePointer, {6, kStorageOutput, 4});
        append_inst(ps_w, kOpTypePointer, {26, kStorageInput, 25});
        append_inst(ps_w, kOpConstantNull, {4, 7});
        append_inst(ps_w, kOpVariable, {6, 8, kStorageOutput});
        for (std::uint32_t v = 11; v <= 24; ++v) append_inst(ps_w, kOpVariable, {5, v, kStorageInput});
        append_inst(ps_w, kOpVariable, {26, 27, kStorageInput});
        append_inst(ps_w, kOpFunction, {1, 9, 0, 2});
        append_inst(ps_w, 248, {10});
        append_inst(ps_w, kOpStore, {8, 7});
        append_inst(ps_w, kOpReturn, {});
        append_inst(ps_w, kOpFunctionEnd, {});

        ShaderEntry vs_e, ps_e;
        ps_e.f16_halves = {{0, {14, 15}}};
        std::string err;
        CHECK(link_f16_halves(vs_w, vs_e, ps_w, ps_e, &err));

        std::map<std::uint32_t, std::uint32_t> vs_locs, ps_locs;
        int vs_u = 0, vs_p = 0, ps_u = 0, ps_p = 0;
        std::set<std::uint32_t> vs_ep, ps_ep_set;
        CHECK(verify_spirv_structure(vs_w, kStorageOutput, vs_locs, vs_u, vs_p, vs_ep));
        CHECK(verify_spirv_structure(ps_w, kStorageInput, ps_locs, ps_u, ps_p, ps_ep_set));
        CHECK(vs_u == 4);
        CHECK(vs_locs.count(24) == 1 && vs_locs.count(25) == 1);
        CHECK(ps_locs.count(24) == 1 && ps_locs.count(25) == 1);
        CHECK(!ps_locs.count(14) && !ps_locs.count(15));
    }

    std::printf("spirv_f16_halves_test: all checks passed\n");
    return 0;
}
