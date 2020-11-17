import numpy


class Data:
    """Data processor for BERT and RNN model for SMP-CAIL2020-Argmine.

    Attributes:
        model_type: 'bert' or 'rnn'
        max_seq_len: int, default: 512
        tokenizer:  BertTokenizer for bert
                    Tokenizer for rnn
    """
    def __init__(self,
                 data_file=''):
        """Initialize data processor for SMP-CAIL2020-Argmine.

        Args:
            vocab_file: one word each line
            max_seq_len: max sequence length, default: 512
            model_type: 'bert' or 'rnn'
                If model_type == 'bert', use BertTokenizer as tokenizer
                Otherwise, use Tokenizer as tokenizer
        """
        self.data_file = data_file


    def load_file(self, file_path=None) -> numpy.ndarray:
        """Load SMP-CAIL2020-Argmine train file and construct TensorDataset.

        Args:
            file_path: train file with last column as label
            train:
                If True, train file with last column as label
                Otherwise, test file without last column as label

        Returns:

        """
        if file_path:
            self.data_file = file_path
        matrix = numpy.load(self.data_file)['matrix']
        return matrix



def test_data():
    """Test for data module."""
    # For BERT model
    data = Data('data/train.npz')
    mat = data.load_file()
    print(mat.shape)



if __name__ == '__main__':
    test_data()