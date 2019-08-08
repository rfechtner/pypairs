import click
from pypairs.tools.cyclone import cyclone, settings, logg
import pandas as pd


@click.command()
@click.option('-d', '--data', required=True,
              type=click.Path(exists=True, file_okay=True, readable=True, resolve_path=True),
              help="Path to the data matrix where rows correspond to cells and columns to genes.")
@click.option('-s', '--sep', default=",", show_default=True, help="Seperator used for data")
@click.option('-t', '--transpose', default=False, is_flag=True, show_default=True, help="Set flag to transpose matrix")
@click.option('-o', '--out', type=click.Path(resolve_path=True, writable=True), help="Store score to csv")
@click.option('-m', '--marker_pairs', type=click.Path(exists=True, file_okay=True, readable=True, resolve_path=True),
              help='Path to marker pairs .json. If not provided default marker pairs are used')
@click.option('-i', '--iterations', default=1000, show_default=True,
              help="An integer specifying the number of iterations for random sampling to obtain a cycle score.")
@click.option('-x', '--min_iter', default=100, show_default=True,
              help="An integer specifying the minimum number of iterations for score estimation.")
@click.option('-y', '--min_pairs', default=50, show_default=True,
              help="An integer specifying the minimum number of pairs for cycle estimation.")
@click.option('-q', '--quantile_transform', default=False, is_flag=True,
              help='Set flag to apply quantile transformation to data')
@click.option('-v', '--verbosity', default=3, show_default=True,
              help="Set level of verbosity from 1 to 4. Where 1 is minimal and 4 most")
def main(data, marker_pairs, iterations, min_iter, min_pairs, quantile_transform, out, sep, transpose, verbosity):
    data_mat = pd.read_csv(data, index_col=0, sep=sep)
    if transpose:
        data_mat = data_mat.T

    settings.verbosity = verbosity

    scores = cyclone(
        data=data_mat, marker_pairs=marker_pairs, iterations = iterations,
        min_iter=min_iter, min_pairs=min_pairs, quantile_transform=quantile_transform
    )

    if out is not None:
        try:
            scores.to_csv(out)
            logg.hint("Written scores to {}".format(out))
        except IOError:
            logg.error("Could not write score to {}".format(out))
    else:
        print(scores)


if __name__ == "__main__":
    main()
