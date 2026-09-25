// bngsim/include/bngsim/net_file_loader.hpp — .net file parser
//
// Defines the loader interface and the BioNetGen .net implementation.
// NetFileLoader parses .net files and routes construction through ModelBuilder.

#pragma once

#include "bngsim/model.hpp"
#include "bngsim/types.hpp"

#include <string>
#include <utility>
#include <vector>

namespace bngsim {

// ─── What the loader reads out of a .net file ────────────────────────────────
//
// The loader works in two phases: it reads the file into these records, then
// feeds them to ModelBuilder. The records are public so a caller can take the
// loader's reading of a file without the build: `bngsim.parse_net_file` returns
// them as a dict, and `build_model_from_parsed` feeds them to the same builder
// the way `NetFileLoader::load` does (issue #803). There is one reading of the
// format, this one.

struct ParsedParam {
    std::string name;
    /// The numeric parse of the value column. Meaningful only when
    /// `is_expression` is false: build() evaluates an expression.
    double value;
    std::string expression;
    bool is_expression; ///< The value column is not a number the numeric parse consumes whole.
};

struct ParsedSpecies {
    std::string name;     ///< With a clamp `$` removed; `fixed` records it.
    double concentration; ///< Re-resolved by build() when `is_param_ref`.
    bool fixed;
    bool is_param_ref;          ///< IC references a parameter name
    std::string param_ref_name; ///< parameter name for IC
};

struct ParsedFunction {
    std::string name;
    std::string expression; ///< As written, `tfun(...)` calls included.
};

struct ParsedObservable {
    std::string name;
    // Entries: (1-based species index, factor)
    std::vector<std::pair<int, double>> entries;
};

struct ParsedReaction {
    std::string comment;
    double stat_factor;
    std::vector<int> reactant_indices_1based; // 1-based species indices
    std::vector<int> product_indices_1based;  // 1-based species indices
    RateLawType type;
    std::string rate_law_name; // param or function name
    std::string legacy_rate_law_type;
    std::vector<std::string> legacy_rate_law_constants;
    // For MM: kcat_name, km_name
    std::string mm_kcat_name;
    std::string mm_km_name;
};

/// Everything `NetFileLoader::load` hands ModelBuilder, in the order it does.
struct NetFileStructure {
    std::vector<ParsedParam> params; ///< Declared, then any lifted `_InitialConc<N>`.
    std::vector<ParsedSpecies> species;
    std::vector<ParsedFunction> functions; ///< Declared, then the Sat/Hill rewrites.
    std::vector<ParsedObservable> observables;
    std::vector<ParsedReaction> reactions; ///< Sat/Hill already rewritten to Functional.
    std::string net_file_dir;              ///< What a relative `tfun('...')` path resolves against.
    std::vector<std::string> load_warnings;
};

/// Read a `.net` file the way `NetFileLoader::load` does, without building it:
/// every block parsed, expression-valued initial concentrations lifted into
/// `_InitialConc<N>` parameters, and the deprecated Sat/Hill rate laws rewritten
/// into explicit functions and observables.
NetFileStructure parse_net_file_structure(const std::string &path);

/// Build what `parse_net_file_structure` read: phase 2 of `NetFileLoader::load`,
/// which is this plus the load warnings.
NetworkModel build_net_file_structure(const NetFileStructure &parsed);

// ─── .net table functions ────────────────────────────────────────────────────

/// One table function a `.net` functions line asks for, read out of the
/// `tfun(...)` call in its expression.
struct NetTableFunction {
    /// Runtime identifier — ExprTk calls the table `tfun_<name>()`. It is the
    /// BNG function's own name when the expression is nothing but a `tfun(...)`
    /// call, and a synthetic `<bng_func>__tfun<k>` when the call sits inside
    /// surrounding arithmetic.
    std::string name;
    /// `.tfun` column-2 header to accept; empty means `name`. Set to the BNG
    /// function name for a synthetic table, whose file still labels its value
    /// column by the name the modeller wrote.
    std::string header_name;
    /// File-based spec: a path relative to the `.net` file's own directory, or
    /// an absolute one. Empty when `is_inline`.
    std::string filepath;
    std::vector<double> xs; ///< Inline data — `tfun([xs],[ys],index)`.
    std::vector<double> ys;
    std::string index_name; ///< "time", or a parameter/observable name.
    std::string method;     ///< "linear" or "step".
    bool is_inline = false;
};

/// What a `.net` functions line becomes once its `tfun(...)` calls are lifted
/// out of the expression.
struct NetFunctionTables {
    /// The expression to hand `ModelBuilder::add_function`. Unchanged when the
    /// line names no table; the whole `tfun(...)` body when the line is one
    /// (`build()` rewrites that to `tfun_<name>()` once the table is loaded);
    /// otherwise the original with each call replaced by `tfun_<synthetic>()`,
    /// so the arithmetic around it survives.
    std::string expression;
    /// The tables to register with the builder before it compiles `expression`.
    std::vector<NetTableFunction> tables;
};

/// Read the `tfun(...)` calls out of one `.net` functions line.
///
/// This is the loader's own post-parse step, reachable on its own so a caller
/// that builds a parsed `.net` itself — `bngsim.build_model_from_parsed`, which
/// builds through the same `ModelBuilder` — registers the same tables from the
/// same spec parse rather than growing a second reading of `tfun(...)` syntax to
/// disagree with this one (issue #597).
///
/// @param func_name  The BNG function's name, which names the table it becomes
///                   and validates the `.tfun` header of the ones it contains.
/// @param expression The functions line's expression, as parsed.
NetFunctionTables net_function_tables(const std::string &func_name, const std::string &expression);

// ─── Abstract loader interface ───────────────────────────────────────────────
class ModelLoader {
  public:
    virtual ~ModelLoader() = default;
    virtual NetworkModel load(const std::string &source) = 0;
};

// ─── .net file loader ────────────────────────────────────────────────────────
class NetFileLoader : public ModelLoader {
  public:
    NetworkModel load(const std::string &path) override;
};

} // namespace bngsim
