"""Periodic-table Δ-gauge figure via pymatviz.

Colors each tested element by its Δ against the WIEN2k all-electron reference
(gradwave's ``delta_wien2k``), so the whole periodic table shows reproducibility
at a glance. The color scale is capped at 1 meV/atom, the band that separates
mature codes, so anything at or below reads as reproducible and the known
outliers (Cu) saturate at the top.

Reads results/delta_summary.json and writes results/delta_ptable.html (always,
no extra dependencies) and results/delta_ptable.png (best effort). The static PNG
needs a Plotly image backend, kaleido>=1 with Chrome, most easily obtained on
NixOS with `nix shell nixpkgs#chromium` around this command. The interactive HTML
needs neither. See gradwave.io.viz.ptable_delta.
"""

from pathlib import Path

from gradwave.io import viz

SP = Path(__file__).parent


def main():
    summary = SP / "results" / "delta_summary.json"
    fig = viz.ptable_delta(
        summary,
        cscale_range=(0.0, 1.0),
        colorbar=dict(title="Δ vs WIEN2k<br>(meV/atom)"),
    )

    html = SP / "results" / "delta_ptable.html"
    fig.write_html(html, include_plotlyjs="cdn")
    print(f"wrote {html}")

    png = SP / "results" / "delta_ptable.png"
    try:
        fig.write_image(png, width=1100, height=640, scale=2)
        print(f"wrote {png}")
    except Exception as err:  # noqa: BLE001 - static export is optional
        print(
            f"skipped PNG ({type(err).__name__}): install a Plotly image backend "
            "(kaleido>=1 + Chrome, e.g. `nix shell nixpkgs#chromium`)"
        )


if __name__ == "__main__":
    main()
