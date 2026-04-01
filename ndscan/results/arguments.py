"""
Functions for pretty-printing user argument data (scan parameters, overrides, …) for
FragmentScanExperiments from ARTIQ results.
"""

from collections.abc import Iterable
from typing import Any

from sipyco import pyon

from ..utils import PARAMS_ARG_KEY


def extract_param_schema(arguments: dict[str, Any]) -> dict[str, Any]:
    """Extract ndscan parameter data from the given ARTIQ arguments directory.

    :param arguments: The arguments for an ARTIQ experiment, as e.g. obtained using
        ``oitg.results.load_hdf5_file(…)["expid"]["arguments"]``.
    """
    try:
        string = arguments[PARAMS_ARG_KEY]
    except KeyError:
        raise KeyError(f"ndscan argument ({PARAMS_ARG_KEY}) not found")

    return pyon.decode(string)


def format_numeric(value, spec: dict[str, Any]) -> str:
    unit = spec.get("unit", "")
    if not unit:
        return str(value)
    return f"{value / spec['scale']} {unit}"


def dump_overrides(schema: dict[str, Any]) -> Iterable[str]:
    """Format information about overrides as a human-readable string.

    :return: Generator yielding the output line-by-line.
    """
    for fqn, overrides in schema["overrides"].items():
        for override in overrides:
            ps = schema["schemata"][fqn]
            value = format_numeric(override["value"], ps["spec"])
            yield f" - {ps['description']}: {value}"
            path = override["path"] or "*"
            yield f"   ({fqn}@{path})"


def format_scan_range(
    typ: str, rang: dict[str, Any], param_spec: dict[str, Any]
) -> str:
    if typ == "linear":
        start = format_numeric(rang["start"], param_spec["spec"])
        stop = format_numeric(rang["stop"], param_spec["spec"])
        return f"{start} to {stop}, {rang['num_points']} points"
    if typ == "refining":
        lower = format_numeric(rang["lower"], param_spec["spec"])
        upper = format_numeric(rang["upper"], param_spec["spec"])
        return f"{lower} to {upper}, refining"
    if typ == "list":
        return f"list: [{rang['values']}]"

    return f"<Unknown scan type '{typ}'.>"


def dump_scan(schema: dict[str, Any]) -> Iterable[str]:
    """Format information about the configured scan (if any) as a human-readable string.

    :return: Generator yielding the output line-by-line.
    """

    if "scan" not in schema:
        yield "No scan information present"
        return

    scan = schema["scan"]

    axes = scan["axes"]
    if not axes:
        yield f"No scan (mode: {scan['no_axes_mode']})"
        return

    yield " - Axes:"
    for ax in axes:
        fqn = ax["fqn"]
        ps = schema["schemata"][fqn]
        path = ax["path"] or "*"
        yield f"   - {ps['description']} ({fqn}@{path}):"
        yield f"     {format_scan_range(ax['type'], ax['range'], ps)}"
    yield f" - Number of repeats of scan: {scan['num_repeats']}"
    yield f" - Number of repeats per point: {scan.get('num_repeats_per_point', 1)}"
    yield f" - Randomise order globally: {scan['randomise_order_globally']}"


def dump_optimise(schema: dict[str, Any]) -> Iterable[str]:
    optimise = schema.get("optimise", {})
    parameters = optimise.get("parameters", [])
    if not parameters:
        yield "No optimisation parameters configured"
        return

    yield " - Parameters:"
    for parameter in parameters:
        fqn = parameter["fqn"]
        ps = schema["schemata"][fqn]
        path = parameter["path"] or "*"
        yield f"   - {ps['description']} ({fqn}@{path}):"
        yield (
            "     min="
            + format_numeric(parameter["min"], ps["spec"])
            + ", initial="
            + format_numeric(parameter["initial"], ps["spec"])
            + ", max="
            + format_numeric(parameter["max"], ps["spec"])
        )

    objective = optimise.get("objective", {})
    yield f" - Objective channel: {objective.get('channel', '')}"
    yield f" - Objective direction: {objective.get('direction', 'min')}"

    algorithm = optimise.get("algorithm", {})
    yield f" - Algorithm: {algorithm.get('kind', 'nelder_mead')}"
    yield f" - Max evaluations: {algorithm.get('max_evals', 100)}"
    yield f" - xatol (fraction of span): {algorithm.get('xatol', 1e-3)}"
    yield f" - fatol: {algorithm.get('fatol', 1e-3)}"
    yield (
        " - Repeats per optimiser point: "
        + str(optimise.get("num_repeats_per_point", 1))
    )
    yield (
        " - Averaging method: "
        + str(optimise.get("averaging_method", "mean"))
    )
    yield (
        " - Give point np.inf cost if transitory errors persist: "
        + str(optimise.get("skip_on_persistent_transitory_error", False))
    )


def dump_vanilla_artiq_args(arguments: dict[str, Any]) -> Iterable[str]:
    """Format all non-ndscan (vanilla ARTIQ) arguments for printing as a list."""
    for name, value in arguments.items():
        if name == PARAMS_ARG_KEY:
            # Skip the argument we serialise our data into.
            continue
        yield f" - {name}: {pyon.encode(value)}"


def summarise(schema: dict[str, Any]) -> str:
    """Convenience method returning a combination of :meth:`dump_overrides` and
    :meth:`dump_scan` ready to be printed.
    """
    result = ""

    execution_mode = schema.get("execution_mode", "scan")
    section_title = "Optimise settings" if execution_mode == "optimise" else "Scan settings"
    result += section_title + "\n"
    result += "=" * len(section_title) + "\n"
    result += "\n"
    if execution_mode == "optimise":
        lines = dump_optimise(schema)
    else:
        lines = dump_scan(schema)
    for s in lines:
        result += s + "\n"
    result += "\n"

    result += "\n"

    result += "Overrides\n"
    result += "=========\n"
    result += "\n"
    for s in dump_overrides(schema):
        result += s + "\n"
    result += "\n"

    return result
