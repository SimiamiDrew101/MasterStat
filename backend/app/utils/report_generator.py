"""
PDF Report Generator for MasterStat
Provides reusable functions for generating PDF reports with tables, plots, and statistics
"""

from datetime import datetime
from io import BytesIO
import base64

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, A4
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    PageBreak, Image, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT


class PDFReportGenerator:
    """Generate professional PDF reports for statistical analyses"""

    def __init__(self, title="Statistical Analysis Report", author="MasterStat"):
        self.title = title
        self.author = author
        self.buffer = BytesIO()
        self.doc = SimpleDocTemplate(
            self.buffer,
            pagesize=letter,
            rightMargin=0.75*inch,
            leftMargin=0.75*inch,
            topMargin=1*inch,
            bottomMargin=0.75*inch
        )
        self.styles = getSampleStyleSheet()
        self._setup_custom_styles()
        self.story = []

    def _setup_custom_styles(self):
        """Setup custom paragraph styles"""
        # Title style
        self.styles.add(ParagraphStyle(
            name='CustomTitle',
            parent=self.styles['Heading1'],
            fontSize=24,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=30,
            alignment=TA_CENTER
        ))

        # Subtitle style
        self.styles.add(ParagraphStyle(
            name='CustomSubtitle',
            parent=self.styles['Heading2'],
            fontSize=14,
            textColor=colors.HexColor('#475569'),
            spaceAfter=12,
            alignment=TA_CENTER
        ))

        # Section heading
        self.styles.add(ParagraphStyle(
            name='SectionHeading',
            parent=self.styles['Heading2'],
            fontSize=16,
            textColor=colors.HexColor('#0f172a'),
            spaceAfter=12,
            spaceBefore=20,
            borderWidth=0,
            borderColor=colors.HexColor('#cbd5e1'),
            borderPadding=6
        ))

        # Subsection heading
        self.styles.add(ParagraphStyle(
            name='SubsectionHeading',
            parent=self.styles['Heading3'],
            fontSize=13,
            textColor=colors.HexColor('#334155'),
            spaceAfter=8,
            spaceBefore=12
        ))

    def add_cover_page(self, subtitle=None, metadata=None):
        """Add a cover page with title, subtitle, and metadata"""
        # Title
        self.story.append(Spacer(1, 2*inch))
        self.story.append(Paragraph(self.title, self.styles['CustomTitle']))

        if subtitle:
            self.story.append(Paragraph(subtitle, self.styles['CustomSubtitle']))

        # Metadata
        self.story.append(Spacer(1, 1.5*inch))
        if metadata:
            data = [[k, v] for k, v in metadata.items()]
            table = Table(data, colWidths=[2.5*inch, 3.5*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f1f5f9')),
                ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1e293b')),
                ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
                ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
                ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 0), (-1, -1), 11),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
            ]))
            self.story.append(table)

        # Date
        self.story.append(Spacer(1, 1*inch))
        date_str = datetime.now().strftime("%B %d, %Y at %I:%M %p")
        self.story.append(Paragraph(
            f"<para align=center>Generated: {date_str}</para>",
            self.styles['Normal']
        ))

        self.story.append(PageBreak())

    def add_section(self, title):
        """Add a section heading"""
        self.story.append(Paragraph(title, self.styles['SectionHeading']))
        self.story.append(Spacer(1, 0.2*inch))

    def add_subsection(self, title):
        """Add a subsection heading"""
        self.story.append(Paragraph(title, self.styles['SubsectionHeading']))
        self.story.append(Spacer(1, 0.1*inch))

    def add_paragraph(self, text):
        """Add a paragraph of text"""
        self.story.append(Paragraph(text, self.styles['Normal']))
        self.story.append(Spacer(1, 0.15*inch))

    def add_table(self, data, headers=None, title=None, col_widths=None):
        """
        Add a formatted table

        Args:
            data: List of lists containing table data
            headers: Optional list of column headers
            title: Optional table title
            col_widths: Optional list of column widths
        """
        if title:
            self.add_subsection(title)

        table_data = []
        if headers:
            table_data.append(headers)
        table_data.extend(data)

        # Auto-calculate column widths if not provided
        if not col_widths:
            available_width = 6.5*inch
            num_cols = len(table_data[0])
            col_widths = [available_width / num_cols] * num_cols

        table = Table(table_data, colWidths=col_widths, repeatRows=1 if headers else 0)

        # Apply styling
        style = [
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 9),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ROWBACKGROUNDS', (0, 1 if headers else 0), (-1, -1),
             [colors.white, colors.HexColor('#f8fafc')])
        ]

        if headers:
            style.extend([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#334155')),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
            ])

        table.setStyle(TableStyle(style))
        self.story.append(table)
        self.story.append(Spacer(1, 0.3*inch))

    def add_summary_stats(self, stats_dict, title="Summary Statistics"):
        """Add a summary statistics section"""
        self.add_subsection(title)

        data = [[k, str(v)] for k, v in stats_dict.items()]
        table = Table(data, colWidths=[3*inch, 3.5*inch])
        table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (0, -1), colors.HexColor('#f1f5f9')),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.HexColor('#1e293b')),
            ('ALIGN', (0, 0), (0, -1), 'LEFT'),
            ('ALIGN', (1, 0), (1, -1), 'RIGHT'),
            ('FONTNAME', (0, 0), (0, -1), 'Helvetica-Bold'),
            ('FONTNAME', (1, 0), (1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
            ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.white, colors.HexColor('#f8fafc')])
        ]))
        self.story.append(table)
        self.story.append(Spacer(1, 0.3*inch))

    def add_image_from_base64(self, image_base64, width=5*inch, height=3.5*inch, title=None):
        """
        Add an image from base64 encoded string

        Args:
            image_base64: Base64 encoded image string
            width: Image width
            height: Image height
            title: Optional image title
        """
        if title:
            self.add_subsection(title)

        try:
            # Decode base64 to bytes
            if ',' in image_base64:
                image_base64 = image_base64.split(',')[1]
            image_data = base64.b64decode(image_base64)
            img_buffer = BytesIO(image_data)

            # Create image
            img = Image(img_buffer, width=width, height=height)
            self.story.append(img)
            self.story.append(Spacer(1, 0.3*inch))
        except Exception as e:
            self.add_paragraph(f"<i>Error loading image: {str(e)}</i>")

    def add_recommendations(self, recommendations):
        """Add a recommendations section"""
        self.add_section("Recommendations")

        for i, rec in enumerate(recommendations, 1):
            self.story.append(Paragraph(
                f"{i}. {rec}",
                self.styles['Normal']
            ))
            self.story.append(Spacer(1, 0.1*inch))

    def add_spacer(self, height=0.3*inch):
        """Add vertical spacing"""
        self.story.append(Spacer(1, height))

    def add_page_break(self):
        """Add a page break"""
        self.story.append(PageBreak())

    def build(self):
        """Build the PDF and return bytes"""
        self.doc.build(self.story)
        self.buffer.seek(0)
        return self.buffer.getvalue()


def format_pvalue(p):
    """Format p-value for display"""
    if p < 0.0001:
        return "<0.0001"
    elif p < 0.001:
        return f"{p:.4f}"
    else:
        return f"{p:.4f}"


def format_number(num, decimals=4):
    """Format number with specified decimal places"""
    if num is None:
        return "N/A"
    try:
        return f"{float(num):.{decimals}f}"
    except (ValueError, TypeError):
        return str(num)
