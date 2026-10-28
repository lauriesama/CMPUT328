from reportlab.lib.pagesizes import letter
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.enums import TA_CENTER
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Image, Spacer, ListFlowable, ListItem
from PIL import Image as PILImage

A = '/home/laurie/CMPUT328/Assignment3/report_assets/'
body = ParagraphStyle('b', fontName='Times-Roman', fontSize=9, leading=10.6, spaceAfter=3)
title = ParagraphStyle('t', fontName='Times-Bold', fontSize=15, leading=18, alignment=TA_CENTER)
sub = ParagraphStyle('s', fontName='Times-Roman', fontSize=11, leading=13, alignment=TA_CENTER)
auth = ParagraphStyle('a', fontName='Times-Roman', fontSize=10, leading=12, alignment=TA_CENTER, spaceAfter=4)
h = ParagraphStyle('h', fontName='Times-Bold', fontSize=11, leading=13, spaceBefore=4, spaceAfter=2)
cap = ParagraphStyle('c', fontName='Times-Italic', fontSize=8, leading=9.5, alignment=TA_CENTER, spaceAfter=3)
cell = ParagraphStyle('cell', fontName='Times-Roman', fontSize=8, leading=9)
cellb = ParagraphStyle('cellb', parent=cell, fontName='Times-Bold')

def P(t, s=body):
    return Paragraph(t, s)

def table(rows, widths, bold_rows=()):
    data = [[Paragraph(str(c), cellb if (r == 0 or r in bold_rows) else cell) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, hAlign='CENTER')
    t.setStyle(TableStyle([
        ('LINEABOVE', (0, 0), (-1, 0), 0.8, colors.black), ('LINEBELOW', (0, 0), (-1, 0), 0.4, colors.black),
        ('LINEBELOW', (0, -1), (-1, -1), 0.8, colors.black),
        ('TOPPADDING', (0, 0), (-1, -1), 1.2), ('BOTTOMPADDING', (0, 0), (-1, -1), 1.2),
        ('LEFTPADDING', (0, 0), (-1, -1), 3), ('RIGHTPADDING', (0, 0), (-1, -1), 3)]))
    return t

def fig(name, width):
    w, hgt = PILImage.open(A + name).size
    return Image(A + name, width=width, height=width * hgt / w)

W = 7.0 * inch
doc = SimpleDocTemplate('/home/laurie/CMPUT328/Assignment3/report.pdf', pagesize=letter,
                        leftMargin=0.75 * inch, rightMargin=0.75 * inch, topMargin=0.55 * inch, bottomMargin=0.5 * inch,
                        title='CMPUT 328 Assignment 3 Report', author='Laurie Gantuangco')
s = []
s += [P('CMPUT 328 Assignment 3', title), P('BDD100K Detection and Instance Segmentation: Day vs. Night', sub),
      P('Laurie Gantuangco &nbsp;|&nbsp; Student ID: 1771419 &nbsp;|&nbsp; October 2026', auth)]

s.append(P('1. AP and Operating-Point Tables', h))
s.append(P('<b>Key takeaway:</b> The from-scratch FCOS-style detector (6.79M parameters) and a U-Net mask network (3.35M) clear every target '
           '(box AP &ge; 14 / AP50 &ge; 30; mask AP &ge; 7.5 / AP50 &ge; 16) and are within a few points of pretrained YOLO26m. '
           'All numbers are on the full BDD100K validation set (10,000 images for boxes; the 454-image segmentation validation set for masks, 377 day / 77 night).'))
t1 = [['System and AP type', 'Full AP', 'AP50', 'Day AP', 'Night AP', 'Day&minus;Night (pp)'],
      ['Student detector &mdash; box', '24.69', '48.40', '25.58', '22.55', '3.04'],
      ['YOLO26m detector &mdash; box', '28.53', '52.70', '30.74', '23.00', '7.74'],
      ['Improved (lighting-aug.) detector &mdash; box', '25.37', '49.28', '26.28', '23.02', '3.26'],
      ['Student pipeline &mdash; mask', '16.14', '38.55', '16.28', '14.65', '1.63'],
      ['YOLO26m-seg &mdash; mask', '18.79', '45.78', '18.82', '16.76', '2.06'],
      ['Mask net on GT boxes (oracle) &mdash; mask', '19.53', '54.75', '19.43', '21.80', '&minus;2.37']]
s.append(table(t1, [2.7 * inch, 0.7 * inch, 0.7 * inch, 0.7 * inch, 0.75 * inch, 1.1 * inch]))
s.append(P('Table 1. AP (%) by split. The student segmentation pipeline uses the improved detector plus the mask network.', cap))
t2 = [['Condition / Split', 'Threshold', 'Precision', 'Recall', 'F1', 'Predictions'],
      ['Full validation', '0.25', '0.3022', '0.7370', '0.4287', '286,320'],
      ['<b>Full validation (chosen: max F1)</b>', '<b>0.50</b>', '0.8437', '0.5551', '0.6696', '77,253'],
      ['Full validation', '0.75', '0.9880', '0.2425', '0.3895', '28,826'],
      ['Daytime (chosen)', '0.50', '0.8640', '0.5536', '0.6748', '43,840'],
      ['Nighttime (chosen)', '0.50', '0.8062', '0.5579', '0.6594', '27,047']]
s.append(table(t2, [2.7 * inch, 0.7 * inch, 0.75 * inch, 0.7 * inch, 0.7 * inch, 0.85 * inch]))
s.append(P('Table 2. Precision/recall of the baseline detector (117,424 ground-truth boxes overall; 68,428 day, 39,086 night). '
           'The threshold 0.50 was chosen as the one with the highest F1 on the full validation set.', cap))

s.append(P('2. Day vs. Night Comparison', h))
s.append(P('<b>Key takeaway:</b> At night my detector loses precision, not recall, and the day&ndash;night AP gap is small (3.0 points). '
           'High-confidence detections stay precise (0.988 at threshold 0.75) because a score that high needs a clearly visible object with strong evidence in both the class '
           'and centerness heads; those cases are almost always correct whatever the lighting. The cost is recall: only 24% of ground-truth boxes are found at 0.75, 56% at 0.50. '
           'At the chosen threshold, recall is the same by day and night (0.554 vs. 0.558) but nighttime precision is 6 points lower (0.806 vs. 0.864), so night adds false alarms '
           '(glare, dark shapes and overlapping vehicles scored as objects; bottom-right of Figure 1) and slightly looser boxes (AP50 drops 3.2 points, AP 3.0). '
           'The missed objects, in both conditions (middle column of Figure 1), are mostly small, distant and crowded vehicles or pedestrians. '
           'I did not compute separate small/medium/large AP, so I cannot say how the drop splits by object size beyond this visual evidence.'))
s.append(fig('c40_1.png', W))
s.append(P('Figure 1. Baseline detector at threshold 0.5; green = ground truth, red = prediction. Columns: correct, misses, false alarms; rows: day, night.', cap))

s.append(P('3. Nighttime Failure Modes and Intervention', h))
s.append(P('<b>Key takeaway:</b> Brightness/contrast augmentation helped a little at night, but also by day, so it acts as general regularization and does not close the gap. '
           'Night images are under-exposed and low in contrast, with glare from lights, so object boundaries and textures are weak, and even though about 40% of frames are night '
           '(3,929 of 10,000 in validation), the many dark small objects are hard to separate from the background. '
           'I tested the hypothesis that varying brightness (0.55&ndash;1.45) and contrast (0.65&ndash;1.35) during training makes features more robust to exposure. The controlled retrain '
           'changed only that setting (same seed, data, architecture, 15 epochs). Night AP rose 22.55&rarr;23.02 (+0.47) and AP50 45.88&rarr;46.72; day AP also rose 25.58&rarr;26.28 (+0.70), '
           'so the gap went from 3.04 to 3.26. Night precision/recall at 0.50 barely moved (0.806&rarr;0.809, 0.558&rarr;0.558). '
           'The gains come from a single seed and are within what I would expect from run-to-run noise, so they are consistent with, not proof of, the hypothesis. '
           'For segmentation, the detector is the night bottleneck: the same mask network scores 21.8 mask AP at night with ground-truth boxes (19.4 by day) but 14.7 with detected boxes. '
           'For large objects (&ge;1,500 px) the box is missed for 1.8% (night) vs. 2.0% (day), and a good box gets a poor mask (IoU &lt; 0.5) for 4.2% vs. 4.0%, so large objects are handled equally well; '
           'the loss comes from small and dark objects.'))
s.append(fig('c66_1.png', W * 0.82))
s.append(P('Figure 2. Student segmentation pipeline at night; green = ground-truth masks, red = predictions, yellow = object of interest. '
           'Left: missed detection (no box). Right: good box (IoU 0.93) but inaccurate mask (IoU 0.35), caused by an occluding neighbouring car.', cap))

s.append(P('4. Student Model vs. YOLO', h))
s.append(P('On cars the student beats YOLO26m (box AP 40.92 vs. 38.52), although YOLO is 3x larger (21.9M vs. 6.79M parameters) and better overall (28.53 vs. 24.69), '
           'driven by pedestrians (29.13 vs. 21.25) and bicycles (17.95 vs. 11.89). The student is trained on 63,000 BDD100K images whose label conventions it learns exactly: dashcam viewpoint, '
           'very small and heavily occluded cars, and a car/pedestrian/bicycle label set. YOLO was trained on COCO with different annotation rules and its COCO classes must be mapped onto ours, '
           'and it is mostly a daytime, varied-viewpoint dataset. Cars are also 87% of the boxes, so the student sees them far more than any other class. '
           'Domain-specific training is worth it when the deployment domain is narrow and has plenty of labelled data, and when label definitions differ from public datasets; '
           'a pretrained model is better for rare classes and when labelled data or compute are limited. At night YOLO\'s advantage nearly vanishes (23.00 vs. 22.55), consistent with this.'))

s.append(P('5. Class Imbalance and Temporal Leakage', h))
s.append(P('Cars make up 87.4% of training boxes (642,093), pedestrians 11.8% (86,322) and bicycles 0.9% (6,447), about 100:1 car to bicycle. The result is a clear AP ordering '
           '(car 40.9 &gt; pedestrian 21.3 &gt; bicycle 11.9) and nighttime bicycle AP near zero in the segmentation set. I used focal loss and report per-class AP rather than only the mean. '
           'BDD100K frames are sampled from continuous videos, so neighbouring frames are near duplicates: the effective number of independent training examples is much smaller than 63,000, '
           'and a random frame-level split would leak almost-identical scenes between train and validation and inflate AP. I used the provided train/validation splits, which BDD100K builds from different videos, '
           'but I did not verify the sequence separation in this release. One further mild leak: the best-epoch checkpoint was selected on a 1,000-image subset of the validation images that is included in the final 10,000-image evaluation.'))

s.append(P('Key Takeaways', h))
items = ['The student detector reached 24.69 AP / 48.40 AP50 (targets 14 / 30); the pipeline reached 16.14 mask AP / 38.55 AP50 (targets 7.5 / 16).',
         'Night costs precision, not recall: at threshold 0.50 recall is 0.554 (day) vs. 0.558 (night), and precision 0.864 vs. 0.806.',
         'Lighting augmentation gave small gains on both day (+0.70) and night (+0.47) AP; it did not close the gap.',
         'In segmentation the detector, not the mask network, limits night performance (21.8 mask AP with GT boxes vs. 14.7 with detections).',
         'Class imbalance (100:1) and frame correlation limit rare-class AP and the validity of random splits.']
s.append(ListFlowable([ListItem(P(i), leftIndent=10) for i in items], bulletType='bullet', bulletFontSize=6, leftIndent=10, bulletOffsetY=-1))

s.append(P('AI-Assistance Disclosure', h))
s.append(P('GitHub Copilot helped implement and debug notebook code and draft this report. I ran the notebook experiments and the figures and numbers are taken from the notebook outputs.'))

doc.build(s)
