#coding=utf-8

from docx import Document
from aip import AipOcr
import pdfkit
import fitz
import os

pdfpath = 'E:\pdf'
pdfname = '2021.pdf'
path_wk = r'D:/Program Files/wkhtmltopdf/bin/wkhtmltopdf.exe'

""" 你的 APPID AK SK """
APP_ID = '25230547'
API_KEY = 'zgPNf0tnh89Du22XHyuiHI7K'
SECRET_KEY = 'YCZIi0ouwFeRXq8lSI3TdBGmhS3srG8V'


# 将每页pdf转为png格式图片
def pdf_image():
    pdf = fitz.open(pdfpath + os.sep + pdfname)
    for pg in range(0, pdf.pageCount):
        # 获得每一页的对象
        page = pdf[pg]
        trans = fitz.Matrix(1.0, 1.0).preRotate(0)
        # 获得每一页的流对象
        pm = page.getPixmap(matrix=trans, alpha=False)
        # 保存图片
        pm.writePNG(image_path + os.sep + pdfname[:-4] + '_' + '{:0>3d}.png'.format(pg + 1))
    page_range = range(pdf.pageCount)
    pdf.close()
    return page_range


# 将图片中的文字转换为字符串
def read_png_str(page_range):
    # 读取本地图片的函数
    def get_file_content(filePath):
        with open(filePath, 'rb') as fp:
            return fp.read()

    allPngStr = []
    image_list = []
    for page_num in page_range:
        # 读取本地图片
        image = get_file_content(image_path + os.sep + r'{}_{}.png'.format(pdfname[:-4], '%03d' % (page_num + 1)))
        print(image)
        image_list.append(image)

    # 新建一个AipOcr
    client = AipOcr(APP_ID, API_KEY, SECRET_KEY)
    # 可选参数
    options = {}
    options["language_type"] = "CHN_ENG"
    options["detect_direction"] = "false"
    options["detect_language"] = "false"
    options["probability"] = "false"
    for image in image_list:
        # 通用文字识别,得到的是一个dict
        pngjson = client.basicGeneral(image, options)
        pngstr = ''
        for x in pngjson['words_result']:
            pngstr = pngstr + x['words'] + '\n'
        print('正在调用百度接口：第{}个，共{}个'.format(len(allPngStr), len(image_list)))
        allPngStr.append(pngstr)
    return allPngStr


def str2word(allPngStr):
    document = Document()
    for i in allPngStr:
        document.add_paragraph(
            i, style='ListBullet'
        )
        document.save(pdfpath + os.sep + pdfname[:-4] + '.docx')

    print('处理完成')


image_path = pdfpath + os.sep + "image"
if not os.path.exists(image_path):
    os.mkdir(image_path)

range_count = pdf_image()
allPngStr = read_png_str(range_count)
str2word(allPngStr)