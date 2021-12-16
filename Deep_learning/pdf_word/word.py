# from pdf2docx import Converter
# biantai = Converter(r'2021.pdf')
# biantai.convert(r'out2021.docx')
# biantai.close()
import argparse
from pdf2docx import Converter


def main(pdf_file, docx_file):
    cv = Converter(pdf_file)
    cv.convert(docx_file, start=0, end=None)
    cv.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('2021.pdf', type=str)
    parser.add_argument('2021.docx', type=str)
    args = parser.parse_args()
    main(args.pdf_file, args.docx_file)