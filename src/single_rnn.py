import sys
from loguru import logger
from tqdm import tqdm
from helper import Args_Parser
from rnn import RNN
from timeseries import Timeseries


def train_single_rnn(num_epochs, lags, data, hidden_layers, hidden_nodes=None) -> None:
    """
    Traing and Test Single Fully Connected RNN
    """
    rnn = RNN(paths=[], centeroids_clusters=None, lags=lags)
    rnn.build_fully_connected_rnn(
        input_names=data.input_names,
        output_names=data.output_names,
        lags=lags,
        hid_layers=hidden_layers,
        hid_nodes=len(data.input_names),
    )
    logger.info("Single RNN:: Staring Single Fully Connected RNN Training/Testing")
    for i in tqdm(range(num_epochs)):
        rnn.do_epoch(data.train_input, data.train_output)
        if i % 5 == 0:
            logger.info(f"Single RNN:: BP Training MSE: {rnn.total_err}")
    rnn.test_rnn(data.test_input, data.test_output)
    logger.info(f"Single RNN:: Testing MSE: {rnn.fitness}")


if __name__ == "__main__":
    args = Args_Parser(sys.argv)
    logger.remove()
    logger.add(sys.stdout, level=args.term_log_level)
    logger.add(
        f"{args.log_dir}/single_rnn_{args.log_file_name}.log",
        level=args.col_log_level,
    )
    data = Timeseries(
        data_files=args.data_files,
        input_params=args.input_names,
        output_params=args.output_names,
        data_dir=args.data_dir,
    )
    train_single_rnn(num_epochs=2000, lags=5, data=data, hidden_layers=10)
