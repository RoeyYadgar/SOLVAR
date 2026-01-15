"""Command-line interface for solvar commands.

This module provides a unified CLI entry point that dispatches to different
solvar commands: analyze, workflow, and analysis-viewer.
"""

import click

from solvar.commands.analysis_viewer import analysis_viewer_cli
from solvar.commands.analyze import analyze_cli
from solvar.commands.workflow import covar_workflow_cli


@click.group()
@click.version_option()
def cli() -> None:
    """Solvar: A package for covariance estimation in cryo-EM."""
    pass


# Add existing commands to the group
cli.add_command(analyze_cli, name="analyze")
cli.add_command(covar_workflow_cli, name="workflow")
cli.add_command(analysis_viewer_cli, name="analysis-viewer")


def main() -> None:
    """Main entry point for the solvar CLI."""
    cli()


if __name__ == "__main__":
    main()
