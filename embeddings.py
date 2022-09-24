import argparse
import transformers

from fake_news.util import get_embeddings
from fake_news.defaults import DEFAULT_TEXT_DICTS_PATH


if __name__ == "__main__":
    transformers.logging.set_verbosity_error()

    parser = argparse.ArgumentParser()

    parser.add_argument("--stances", type=str, help="path to .csv stance file", required=True)
    parser.add_argument("--text_dicts", type=str, help="path to saved text_dicts .pkl file", default=DEFAULT_TEXT_DICTS_PATH)
    parser.add_argument("--output_file", type=str, help="where to save the embeddings")
    parser.add_argument("--batch_size", type=int, help="inputs for BERT in batches", default=1)
    parser.add_argument("--gpu",  help="whether to use GPU", action='store_true')
    parser.add_argument("--verbose", help="whether to log progress with tqdm", action='store_true')

    args = parser.parse_args()
    output_file = args.output_file

    if output_file is None:
        file_name = args.stances.split('/')[-1][:-4]
        output_file = f"preprocessed/fake_news/embeddings/{file_name}.pkl"

    get_embeddings(
        args.stances, args.text_dicts, output_file, args.batch_size, args.gpu, args.verbose
    )
