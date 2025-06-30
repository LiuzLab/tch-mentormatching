from bin.processing.io_utils import extract_text_from_txt, load_documents, convert_txt_dir_to_csv

def test_extract_text_from_txt():
    text = extract_text_from_txt('test_data/test.txt')
    assert text == "This is a test text file."

def test_load_documents():
    docs = load_documents('test_data', extensions=['.txt'])
    assert len(docs) == 1
    assert docs[0][0] == 'test.txt'
    assert docs[0][1] == 'This is a test text file.'

def test_convert_txt_dir_to_csv():
    convert_txt_dir_to_csv('test_data/*.txt', 'test_data/test.csv')
    df = pd.read_csv('test_data/test.csv')
    assert len(df) == 1
    assert df['filename'][0] == 'test.txt'
    assert df['content'][0] == 'This is a test text file.'
    os.remove('test_data/test.csv')
