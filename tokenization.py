import argparse
import transformers

from fake_news.util import tokenize_texts
from fake_news.defaults import DEFAULT_TEXT_DICTS_PATH


if __name__ == "__main__":
    transformers.logging.set_verbosity_error()

    parser = argparse.ArgumentParser()

    parser.add_argument("--text_dicts", type=str, help="path to saved text_dicts .pkl file", default=DEFAULT_TEXT_DICTS_PATH)
    parser.add_argument("--output_file", type=str, help="where to save new text_dicts file with input ids after tokenization", default=DEFAULT_TEXT_DICTS_PATH)
    parser.add_argument("--verbose", help="whether to log progress with tqdm", action='store_true')

    args = parser.parse_args()

    tokenize_texts(args.text_dicts, args.output_file, args.verbose)
