from pathlib import Path
from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.shared import Pt
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
f=Path('/Users/zaxharmerkurev/Documents/dimplom/статья/исследование_с_иллюстрациями/Merkurev_research_article.docx');d=Document(f)
for root in [d.styles.element,d._element]:
 for el in list(root.iter(qn('w:pBdr'))):el.getparent().remove(el)
 for el in root.iter(qn('w:rFonts')):
  for attr in ['asciiTheme','hAnsiTheme','eastAsiaTheme','cstheme','csTheme']:
   el.attrib.pop(qn('w:'+attr),None)
  for attr in ['ascii','hAnsi','cs','eastAsia']:el.set(qn('w:'+attr),'Times New Roman')
for p in d.paragraphs:
 if p.text.startswith('Меркурьев Захар'):
  p.alignment=WD_ALIGN_PARAGRAPH.LEFT
  p.text='Меркурьев Захар Алексеевич\nНаучный руководитель: Евдокимова Инга Сергеевна\nФГБОУ ВО «Восточно-Сибирский государственный университет технологий и управления»'
 if p.text.startswith('Abstract.'):
  for r in p.runs:r.text=r.text.replace('Thirty eight step routes','Thirty routes of eight steps each')
 if p.text.startswith('В статье готовые реализации'):
  p.text='Для сравнения используются функциональные обозначения: A — статическая модель, A + IOR — её вариант с подавлением возврата, B — последовательная модель человеческого взгляда [2, 3]. Оценка проводится на готовых реализациях. Первичные таблицы, параметры и версии сохраняются отдельно, производные показатели пересчитываются без изменения базы записей.'
 if p.text.startswith('Полученные результаты образуют'):
  for r in p.runs:r.text=r.text.replace('не доказывает точность рекламы','не доказывает точность прогноза на рекламе')
 if p.text.startswith('7. Hussain'):
  p.paragraph_format.keep_together=True
# prevent split cells and keep short tables whole
for i,t in enumerate(d.tables):
 for row in t.rows:
  pr=row._tr.get_or_add_trPr();pr.append(OxmlElement('w:cantSplit'))
 if i in [0,1,2,4,6]:
  for row in t.rows[:-1]:
   for c in row.cells:
    for p in c.paragraphs:p.paragraph_format.keep_with_next=True
# use conventional half-up display consistently with running text
for c,v in zip(d.tables[6].rows[4].cells,['20','3,13','0,63']):c.text=v
# A compact bibliography avoids a nearly empty terminal page.
started=False
for p in d.paragraphs:
 if p.text=='Список литературы':started=True;continue
 if started:
  p.alignment=WD_ALIGN_PARAGRAPH.LEFT;p.paragraph_format.space_after=Pt(4);p.paragraph_format.line_spacing=1.0
  for r in p.runs:r.font.size=Pt(11)
d.save(f)
print(f)
