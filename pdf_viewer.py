# from reportlab.pdfgen import canvas
# from reportlab.lib.pagesizes import letter
# from reportlab.pdfbase import pdfdoc
#
# pdf_file = "example.pdf"
# pdfdoc.PDFViewer(pdf_file)
# from PyPDF2 import PdfReader
#
# reader = PdfReader("example.pdf")
# number_of_pages = len(reader.pages)
# page = reader.pages[10]
# text = page.extract_text()
# print(text)




import webbrowser

pdf_file = "example.pdf"
webbrowser.open_new(pdf_file)