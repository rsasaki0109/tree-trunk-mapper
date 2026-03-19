"""CLI interface for tree-trunk-mapper."""

from __future__ import annotations

from pathlib import Path

import click

import numpy as np

from tree_trunk_mapper import __version__


@click.group()
@click.version_option(version=__version__)
def cli() -> None:
    """Detect and map individual tree trunks from LiDAR point clouds."""


@cli.command()
@click.argument("input_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--slice-height",
    default=1.3,
    show_default=True,
    help="Height above ground for breast-height slice (metres).",
)
@click.option(
    "--slice-thickness",
    default=0.2,
    show_default=True,
    help="Thickness of the height slice (metres).",
)
@click.option(
    "--eps",
    default=0.15,
    show_default=True,
    help="DBSCAN epsilon (metres).",
)
@click.option(
    "--min-samples",
    default=20,
    show_default=True,
    help="DBSCAN minimum samples per cluster.",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for detection results (JSON). Defaults to <input>_detections.json.",
)
def detect(
    input_path: Path,
    slice_height: float,
    slice_thickness: float,
    eps: float,
    min_samples: int,
    output: Path | None,
) -> None:
    """Detect tree trunks in a single point cloud frame."""
    import json

    from tree_trunk_mapper.loader import load_point_cloud
    from tree_trunk_mapper.detector import detect_trunks

    click.echo(f"Loading point cloud from {input_path} ...")
    pcd = load_point_cloud(input_path)

    click.echo("Detecting tree trunks ...")
    detections = detect_trunks(
        pcd,
        slice_height=slice_height,
        slice_thickness=slice_thickness,
        eps=eps,
        min_samples=min_samples,
    )

    if output is None:
        output = input_path.with_name(input_path.stem + "_detections.json")

    results = [d.to_dict() for d in detections]
    output.write_text(json.dumps(results, indent=2))
    click.echo(f"Found {len(detections)} trunk(s). Results saved to {output}")


@cli.command(name="map")
@click.argument("input_dir", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--merge-radius",
    default=0.5,
    show_default=True,
    help="Radius (metres) within which detections are merged into a single trunk.",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for the trunk map (JSON). Defaults to <input_dir>/trunk_map.json.",
)
@click.option(
    "--glob-pattern",
    default="*.pcd",
    show_default=True,
    help="Glob pattern to find point cloud files in the input directory.",
)
def build_map(
    input_dir: Path,
    merge_radius: float,
    output: Path | None,
    glob_pattern: str,
) -> None:
    """Build a persistent trunk map from a sequence of point cloud frames."""
    import json

    from tree_trunk_mapper.loader import load_point_cloud
    from tree_trunk_mapper.detector import detect_trunks
    from tree_trunk_mapper.mapper import TrunkMapper

    files = sorted(input_dir.glob(glob_pattern))
    if not files:
        click.echo(f"No files matching '{glob_pattern}' found in {input_dir}")
        raise SystemExit(1)

    mapper = TrunkMapper(merge_radius=merge_radius)

    with click.progressbar(files, label="Processing frames") as bar:
        for f in bar:
            pcd = load_point_cloud(f)
            detections = detect_trunks(pcd)
            mapper.add_detections(detections)

    trunk_map = mapper.get_map()

    if output is None:
        output = input_dir / "trunk_map.json"

    output.write_text(json.dumps([t.to_dict() for t in trunk_map], indent=2))
    click.echo(f"Mapped {len(trunk_map)} unique trunk(s). Saved to {output}")


@cli.command()
@click.argument("map_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--format", "-f",
    "fmt",
    type=click.Choice(["geojson", "csv"], case_sensitive=False),
    default="geojson",
    show_default=True,
    help="Export format.",
)
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output file path. Defaults to <map_path>.<format>.",
)
def export(map_path: Path, fmt: str, output: Path | None) -> None:
    """Export a trunk map as GeoJSON or CSV."""
    import json

    from tree_trunk_mapper.export import export_geojson, export_csv
    from tree_trunk_mapper.mapper import TrunkRecord

    raw = json.loads(map_path.read_text())

    # Support both detect output (TrunkDetection) and map output (TrunkRecord)
    trunks = []
    for i, d in enumerate(raw):
        if "trunk_id" in d:
            trunks.append(TrunkRecord.from_dict(d))
        else:
            # Convert TrunkDetection format to TrunkRecord
            trunks.append(TrunkRecord(
                trunk_id=i,
                position=np.array(d["center"]),
                dbh=d["dbh"],
                observation_count=1,
            ))

    if output is None:
        ext = "geojson" if fmt == "geojson" else "csv"
        output = map_path.with_suffix(f".{ext}")

    if fmt == "geojson":
        export_geojson(trunks, output)
    else:
        export_csv(trunks, output)

    click.echo(f"Exported {len(trunks)} trunk(s) to {output}")


@cli.command()
@click.argument("input_pcd", type=click.Path(exists=True, path_type=Path))
@click.argument("detections_json", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output PNG path. Defaults to <detections_json>.png.",
)
@click.option(
    "--slice-height",
    default=1.3,
    show_default=True,
    help="Height above ground for the slice indicator (metres).",
)
@click.option(
    "--slice-thickness",
    default=0.2,
    show_default=True,
    help="Thickness of the height slice indicator (metres).",
)
def visualize(
    input_pcd: Path,
    detections_json: Path,
    output: Path | None,
    slice_height: float,
    slice_thickness: float,
) -> None:
    """Visualize detected trunks overlaid on the point cloud.

    Produces a two-panel PNG: top-down view with detected trunk circles (left)
    and a side (XZ) view showing the height slice region (right).
    """
    import json
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle

    from tree_trunk_mapper.loader import load_point_cloud
    from tree_trunk_mapper.detector import TrunkDetection

    # Load data
    pcd = load_point_cloud(input_pcd)
    points = np.asarray(pcd.points)

    raw = json.loads(detections_json.read_text())
    detections = [TrunkDetection.from_dict(d) for d in raw]

    if output is None:
        output = detections_json.with_suffix(".png")

    # Estimate ground level for slice indicator
    ground_z = float(np.percentile(points[:, 2], 5))
    z_lo = ground_z + slice_height - slice_thickness / 2
    z_hi = ground_z + slice_height + slice_thickness / 2

    fig, (ax_top, ax_side) = plt.subplots(1, 2, figsize=(16, 8))

    # --- Left panel: top-down (XY) view ---
    # Subsample points for plotting performance
    max_plot = 50_000
    if len(points) > max_plot:
        idx = np.random.default_rng(0).choice(len(points), max_plot, replace=False)
        plot_pts = points[idx]
    else:
        plot_pts = points

    ax_top.scatter(plot_pts[:, 0], plot_pts[:, 1], s=0.3, c="0.7", alpha=0.4, rasterized=True)
    for det in detections:
        circ = Circle(
            (det.center[0], det.center[1]),
            det.radius,
            fill=False,
            edgecolor="red",
            linewidth=1.5,
        )
        ax_top.add_patch(circ)
        ax_top.plot(det.center[0], det.center[1], "r+", markersize=6)
        ax_top.annotate(
            f"D={det.dbh:.2f}m",
            (det.center[0], det.center[1]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=7,
            color="red",
        )

    ax_top.set_xlabel("X (m)")
    ax_top.set_ylabel("Y (m)")
    ax_top.set_title(f"Top-Down View  ({len(detections)} trunks detected)")
    ax_top.set_aspect("equal")

    # --- Right panel: side (XZ) view ---
    ax_side.scatter(plot_pts[:, 0], plot_pts[:, 2], s=0.3, c="0.7", alpha=0.4, rasterized=True)
    ax_side.axhspan(z_lo, z_hi, color="green", alpha=0.15, label=f"Slice [{z_lo:.2f}, {z_hi:.2f}]m")
    ax_side.axhline(z_lo, color="green", linewidth=0.8, linestyle="--")
    ax_side.axhline(z_hi, color="green", linewidth=0.8, linestyle="--")

    for det in detections:
        ax_side.plot(det.center[0], det.center[2], "r^", markersize=6)

    ax_side.set_xlabel("X (m)")
    ax_side.set_ylabel("Z (m)")
    ax_side.set_title("Side View (XZ) with Height Slice")
    ax_side.legend(loc="upper right", fontsize=8)

    fig.suptitle(f"tree-trunk-mapper: {input_pcd.name}", fontsize=12, fontweight="bold")
    fig.tight_layout()
    fig.savefig(str(output), dpi=150, bbox_inches="tight")
    plt.close(fig)

    click.echo(f"Visualization saved to {output}")
