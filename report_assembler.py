import glob
import json
import os

from fpdf import FPDF


def assemble_report(out_dir):
    with open(out_dir + '/out.json', "r") as f:
        info = f.read()
    pdf = FPDF(format='letter', unit='in')
    pdf.add_page()
    pdf.set_font('Times', '', 8.0)
    epw = pdf.w - 2 * pdf.l_margin

    pdf.set_font('Times', 'B', 12.0)
    pdf.cell(epw, 0.0, 'CT Scan Report', align='C')
    pdf.set_font('Times', '', 8.0)
    pdf.ln(0.3)

    th = pdf.font_size

    data = json.loads(info)

    k_col_width = max([len(k) for k in data.keys()]) * th/1.9
    v_col_width = max([len(v) for v in data.values()]) * th/1.9

    for k, v in data.items():
        pdf.cell(k_col_width, 2*th, k, border=1)
        pdf.cell(v_col_width, 2*th, v, border=1)
        pdf.ln(2*th)

    pdf.ln(0.3)
    pdf.set_font('Times', 'B', 12.0)
    pdf.cell(epw, 0.0, '3D Analysis', align='C')
    pdf.set_font('Times', '', 8.0)
    pdf.ln(0.3)

    pdf.image(out_dir + "/natural.png", x=1, y=5, w=3, h=3)
    pdf.image(out_dir + "/front.png", x=4, y=5, w=3, h=3)
    pdf.image(out_dir + "/top.png", x=1, y=8, w=3, h=3)
    pdf.image(out_dir + "/lateral.png", x=4, y=8, w=3, h=3)

    two_d_slices = glob.glob(out_dir + '/**/*.jpg', recursive=True)

    delta = 3
    for idx in range(0, len(two_d_slices) - 4, 4):
        pdf.add_page()
        pdf.set_font('Times', 'B', 12.0)
        pdf.cell(epw, 0.0, '2D Analysis of Slices : ' +
                 os.path.basename(two_d_slices[idx])[:-4] +
                 ', ' + os.path.basename(two_d_slices[idx + 1])[:-4] +
                 ', ' + os.path.basename(two_d_slices[idx + 2])[:-4] +
                 ', ' + os.path.basename(two_d_slices[idx + 3])[:-4]
                 , align='C')
        pdf.image(two_d_slices[idx], x=1, y=2, w=delta, h=delta)
        pdf.image(two_d_slices[idx + 1], x=delta + 2, y=2, w=delta, h=delta)
        pdf.image(two_d_slices[idx + 2], x=1, y=delta + 4, w=delta, h=delta)
        pdf.image(two_d_slices[idx + 3], x=delta + 2, y=delta + 4, w=delta, h=delta)

    pdf.output(out_dir + "/Report.pdf", 'F')


if __name__ == "__main__":
    out_directory = '1.2.826.0.1.3680043.8.1678.101.10637213214991521358.314450'
    assemble_report(out_directory)
