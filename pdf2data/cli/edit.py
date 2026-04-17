import json

import click

from pdf2data.edit import JsonBoxEditor


@click.command()
@click.argument("input_json", type=click.Path(exists=True, dir_okay=False))
@click.argument("output_json", type=click.Path(dir_okay=False))
@click.option("--kind", type=click.Choice(["block", "tableCell", "tableCaption"]), required=True)
@click.option("--block-index", type=int, required=True)
@click.option("--value", type=str, required=True)
@click.option("--row", type=int, default=None)
@click.option("--col", type=int, default=None)
def edit_json(
    input_json: str,
    output_json: str,
    kind: str,
    block_index: int,
    value: str,
    row: int,
    col: int,
) -> None:
    """Edit a JSON block/cell/caption by target and write the updated JSON."""
    with open(input_json, "r") as f:
        data = json.load(f)

    editor = JsonBoxEditor(data=data)
    target = {"kind": kind, "block_index": block_index}
    if kind == "tableCell":
        if row is None or col is None:
            raise click.UsageError("--row and --col are required for kind=tableCell")
        target["row"] = row
        target["col"] = col

    editor.update_target(target, value)

    with open(output_json, "w") as f:
        json.dump(editor.data, f, indent=4)


def main() -> None:
    edit_json()


if __name__ == "__main__":
    main()
