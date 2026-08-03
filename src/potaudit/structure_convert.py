from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

from ase.data import atomic_numbers
from ase.io import read, write


SUPPORTED_FORMATS = ("poscar", "data", "dump")


@dataclass(frozen=True)
class ConvertReport:
    input_path: Path
    output_path: Path
    input_format: str
    output_format: str
    natoms: int


def _normalize_format(fmt: str) -> str:
    key = fmt.strip().lower()
    aliases = {
        "vasp": "poscar",
        "poscar": "poscar",
        "contcar": "poscar",
        "lmp": "data",
        "lammps": "data",
        "lammps-data": "data",
        "data": "data",
        "dum": "dump",
        "dump": "dump",
        "lammpstrj": "dump",
        "lammps-dump": "dump",
        "lammps-dump-text": "dump",
    }
    if key not in aliases:
        allowed = ", ".join(SUPPORTED_FORMATS)
        raise ValueError(f"Unsupported structure format '{fmt}'. Supported formats: {allowed}")
    return aliases[key]


def _infer_format(path: Path) -> str:
    name = path.name.lower()
    suffix = path.suffix.lower()

    if (
        name in {"poscar", "contcar"}
        or name.startswith(("poscar.", "poscar_", "contcar.", "contcar_"))
        or suffix in {".poscar", ".contcar", ".vasp"}
    ):
        return "poscar"
    if (
        name in {"lmp", "lammps", "data"}
        or name.startswith(("data.", "data_", "lmp.", "lmp_", "lammps.", "lammps_"))
        or suffix in {".lmp", ".lammps", ".data"}
    ):
        return "data"
    if (
        name in {"dum", "dump", "lammpstrj"}
        or name.startswith(("dum.", "dum_", "dump.", "dump_"))
        or suffix in {".dum", ".dump", ".lammpstrj"}
    ):
        return "dump"

    raise ValueError(
        f"Cannot infer structure format from '{path}'. "
        "Use --in-format/--out-format with poscar, data, or dump."
    )


def _ase_format(fmt: str) -> str:
    return {
        "poscar": "vasp",
        "data": "lammps-data",
        "dump": "lammps-dump-text",
    }[fmt]


def _first_seen_symbols(symbols: Iterable[str]) -> List[str]:
    seen = set()
    order: List[str] = []
    for symbol in symbols:
        if symbol not in seen:
            seen.add(symbol)
            order.append(symbol)
    return order


def _parse_lammps_species_symbols(species: Optional[str]) -> Optional[List[str]]:
    if species is None:
        return None

    symbols = [item.strip() for item in species.replace(",", " ").split()]
    if not symbols:
        raise ValueError("--lammps-species must contain at least one element symbol")

    for symbol in symbols:
        if symbol not in atomic_numbers:
            raise ValueError(f"Unknown element symbol in --lammps-species: {symbol}")
    return symbols


def _parse_lammps_z_of_type(species: Optional[str]) -> Optional[dict[int, int]]:
    symbols = _parse_lammps_species_symbols(species)
    if symbols is None:
        return None
    return {idx: atomic_numbers[symbol] for idx, symbol in enumerate(symbols, start=1)}


def read_structure(
    path: str | Path,
    *,
    input_format: Optional[str] = None,
    lammps_species: Optional[str] = None,
    index: int = -1,
):
    in_path = Path(path)
    fmt = _normalize_format(input_format) if input_format else _infer_format(in_path)

    kwargs = {}
    if fmt == "data":
        z_of_type = _parse_lammps_z_of_type(lammps_species)
        if z_of_type is not None:
            kwargs["Z_of_type"] = z_of_type
    elif fmt == "dump":
        kwargs["order"] = False
        kwargs["index"] = index
        specorder = _parse_lammps_species_symbols(lammps_species)
        if specorder is not None:
            kwargs["specorder"] = specorder

    return read(str(in_path), format=_ase_format(fmt), **kwargs), fmt


def write_structure(
    atoms,
    path: str | Path,
    *,
    output_format: Optional[str] = None,
    lammps_style: str = "atomic",
    lammps_units: str = "metal",
) -> str:
    out_path = Path(path)
    fmt = _normalize_format(output_format) if output_format else _infer_format(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if fmt == "poscar":
        write(str(out_path), atoms, format="vasp", vasp5=True, direct=True, sort=False)
    elif fmt == "data":
        specorder = _first_seen_symbols(atoms.get_chemical_symbols())
        write(
            str(out_path),
            atoms,
            format="lammps-data",
            atom_style=lammps_style,
            units=lammps_units,
            masses=True,
            specorder=specorder,
        )
    elif fmt == "dump":
        raise ValueError("LAMMPS dump is supported as input only. Use --out data to write a LAMMPS data file.")
    else:
        raise ValueError(f"Unsupported output format '{fmt}'")

    return fmt


def convert_structure(
    *,
    input_path: str | Path,
    output_path: str | Path,
    input_format: Optional[str] = None,
    output_format: Optional[str] = None,
    lammps_species: Optional[str] = None,
    lammps_style: str = "atomic",
    lammps_units: str = "metal",
    index: int = -1,
) -> ConvertReport:
    atoms, in_fmt = read_structure(
        input_path,
        input_format=input_format,
        lammps_species=lammps_species,
        index=index,
    )
    out_fmt = write_structure(
        atoms,
        output_path,
        output_format=output_format,
        lammps_style=lammps_style,
        lammps_units=lammps_units,
    )

    return ConvertReport(
        input_path=Path(input_path),
        output_path=Path(output_path),
        input_format=in_fmt,
        output_format=out_fmt,
        natoms=len(atoms),
    )


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--in", dest="input_path", required=True, help="Input structure file, e.g. POSCAR, lmp/data, or dump/dum")
    parser.add_argument("--out", dest="output_path", required=True, help="Output structure file, e.g. POSCAR or lmp/data")
    parser.add_argument(
        "--in-format",
        choices=(
            "poscar",
            "vasp",
            "contcar",
            "lmp",
            "lammps",
            "lammps-data",
            "data",
            "dum",
            "dump",
            "lammpstrj",
            "lammps-dump",
            "lammps-dump-text",
        ),
        default=None,
        help="Input format override when it cannot be inferred from the file name",
    )
    parser.add_argument(
        "--out-format",
        choices=("poscar", "vasp", "contcar", "lmp", "lammps", "lammps-data", "data"),
        default=None,
        help="Output format override when it cannot be inferred from the file name",
    )
    parser.add_argument(
        "--lammps-species",
        default=None,
        help="Element order for reading LAMMPS data/dump atom types without reliable masses, e.g. 'Si O' for type 1=Si, type 2=O",
    )
    parser.add_argument("--index", type=int, default=-1, help="Frame index for LAMMPS dump input (default: -1, last frame)")
    parser.add_argument("--lammps-style", default="atomic", help="LAMMPS atom style for data output (default: atomic)")
    parser.add_argument("--lammps-units", default="metal", help="LAMMPS units for data output (default: metal)")


def run(args: argparse.Namespace) -> int:
    report = convert_structure(
        input_path=args.input_path,
        output_path=args.output_path,
        input_format=args.in_format,
        output_format=args.out_format,
        lammps_species=args.lammps_species,
        lammps_style=args.lammps_style,
        lammps_units=args.lammps_units,
        index=args.index,
    )
    print(
        f"[PotAudit] converted {report.input_format} -> {report.output_format} "
        f"natoms={report.natoms} out={report.output_path}"
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Convert structure files between POSCAR and LAMMPS data/dump formats.")
    add_arguments(parser)
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
