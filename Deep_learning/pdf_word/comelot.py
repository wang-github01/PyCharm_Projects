import camelot.io as camelot
data1= camelot.read_pdf("2021.pdf", pages='70',flavor='stream')
data1[0].to_csv('data1.csv',encoding='utf_8_sig')