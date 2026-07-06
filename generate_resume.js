const {
  Document, Packer, Paragraph, TextRun, Table, TableRow, TableCell,
  AlignmentType, LevelFormat, BorderStyle, WidthType, ShadingType,
  TabStopType, TabStopPosition, ExternalHyperlink, UnderlineType
} = require('docx');
const fs = require('fs');
const path = require('path');

// ── COLORS ──────────────────────────────────────────────────────────────────
const BLUE   = "1F4E79";
const ACCENT = "2E75B6";
const GRAY   = "595959";
const BLACK  = "000000";
const WHITE  = "FFFFFF";
const LIGHT  = "EBF3FB";

// ── HELPERS ─────────────────────────────────────────────────────────────────
const txt = (text, opts = {}) => new TextRun({
  text, font: "Calibri", size: opts.size || 20, bold: opts.bold || false,
  color: opts.color || BLACK, italics: opts.italic || false,
  underline: opts.underline ? { type: UnderlineType.SINGLE } : undefined
});

const link = (text, url) => new ExternalHyperlink({
  link: url,
  children: [new TextRun({ text, font: "Calibri", size: 20, color: ACCENT, underline: { type: UnderlineType.SINGLE } })]
});

const para = (children, opts = {}) => new Paragraph({
  children: Array.isArray(children) ? children : [children],
  alignment: opts.align || AlignmentType.LEFT,
  spacing: { before: opts.before || 0, after: opts.after || 60 },
  tabStops: opts.tabStops || [],
  numbering: opts.bullet ? { reference: "bullets", level: 0 } : undefined,
  border: opts.borderBottom ? {
    bottom: { style: BorderStyle.SINGLE, size: 6, color: ACCENT, space: 4 }
  } : undefined,
  indent: opts.indent ? { left: 360 } : undefined,
  shading: opts.shade ? { fill: LIGHT, type: ShadingType.CLEAR } : undefined,
});

const sectionHead = (title) => para(
  [txt(title.toUpperCase(), { bold: true, size: 22, color: BLUE })],
  { before: 160, after: 40, borderBottom: true }
);

const rightTabStop = [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }];

const border0 = { style: BorderStyle.NONE, size: 0, color: "FFFFFF" };
const noBorder = { top: border0, bottom: border0, left: border0, right: border0 };

// ── BULLET CONFIG ────────────────────────────────────────────────────────────
const numbering = {
  config: [{
    reference: "bullets",
    levels: [{
      level: 0, format: LevelFormat.BULLET, text: "\u2022",
      alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 480, hanging: 240 }, spacing: { after: 40 } } }
    }]
  }]
};

// ── DOCUMENT ─────────────────────────────────────────────────────────────────
const doc = new Document({
  numbering,
  styles: {
    default: { document: { run: { font: "Calibri", size: 20 } } }
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 720, right: 1080, bottom: 720, left: 1080 }
      }
    },
    children: [

      // ── NAME ──────────────────────────────────────────────────────────────
      para([txt("VIJAY BEDAGE", { bold: true, size: 36, color: BLUE })],
        { align: AlignmentType.CENTER, after: 20 }),

      para([txt("Data Analyst  |  Machine Learning Enthusiast", { size: 22, color: GRAY, italic: true })],
        { align: AlignmentType.CENTER, after: 60 }),

      // ── CONTACT LINE ──────────────────────────────────────────────────────
      para([
        txt("+91 7795745576", { size: 18, color: GRAY }),
        txt("   |   ", { size: 18, color: GRAY }),
        link("Vijaybedage24@gmail.com", "mailto:Vijaybedage24@gmail.com"),
        txt("   |   ", { size: 18, color: GRAY }),
        link("LinkedIn", "https://www.linkedin.com/in/vijay-bedage-226457315/"),
        txt("   |   ", { size: 18, color: GRAY }),
        link("GitHub: Vijaybedage", "https://github.com/Vijaybedage"),
      ], { align: AlignmentType.CENTER, after: 120 }),

      // ── OBJECTIVE ─────────────────────────────────────────────────────────
      sectionHead("Professional Summary"),
      para([txt(
        "Results-driven Computer Science graduate and MCA student with hands-on experience in Data Analytics, " +
        "Machine Learning, and Business Intelligence. Skilled in Python (Pandas, NumPy, Matplotlib, Seaborn, scikit-learn), " +
        "SQL, Power BI, and Tableau. Completed two internships — Data Analytics at Leosias Technologies and Machine Learning " +
        "at Unified Mentor — with a proven ability to turn complex datasets into actionable insights.",
        { size: 19, color: GRAY }
      )], { after: 80 }),

      // ── SKILLS ────────────────────────────────────────────────────────────
      sectionHead("Technical Skills"),

      para([
        txt("Languages:         ", { bold: true }),
        txt("Python, R, C/C++, SQL"),
      ], { after: 40 }),
      para([
        txt("Libraries:           ", { bold: true }),
        txt("NumPy, Pandas, Matplotlib, Seaborn, scikit-learn, TensorFlow, Keras"),
      ], { after: 40 }),
      para([
        txt("BI & Data Tools:  ", { bold: true }),
        txt("Power BI, Tableau, Advanced Excel, MySQL"),
      ], { after: 40 }),
      para([
        txt("Platforms:           ", { bold: true }),
        txt("Jupyter Notebook, VS Code, Google Colab, Git, GitHub"),
      ], { after: 40 }),
      para([
        txt("Concepts:           ", { bold: true }),
        txt("EDA, Machine Learning, Transfer Learning, ETL, Data Visualization, Agile / SDLC"),
      ], { after: 80 }),

      // ── EXPERIENCE ────────────────────────────────────────────────────────
      sectionHead("Work Experience"),

      // ML Internship
      para([
        txt("Machine Learning Intern  —  Unified Mentor Pvt Ltd", { bold: true, size: 22 }),
        txt("\t"),
        txt("Jun 2025 – Sep 2025", { italic: true, color: GRAY }),
      ], { tabStops: rightTabStop, after: 40 }),

      para([txt("Gained hands-on experience building and deploying ML models using Python and scikit-learn.")],
        { bullet: true }),
      para([txt("Built an Animal Image Classification system using MobileNetV2 Transfer Learning — 15 classes, ~1,944 images, 93%+ validation accuracy.")],
        { bullet: true }),
      para([txt("Developed data preprocessing pipelines, applied data augmentation, and evaluated models using confusion matrices and classification reports.")],
        { bullet: true }),

      para([], { after: 60 }),

      // Data Analytics Internship
      para([
        txt("Data Analytics Intern  —  Leosias Technologies", { bold: true, size: 22 }),
        txt("\t"),
        txt("Oct 2024 – Mar 2025", { italic: true, color: GRAY }),
      ], { tabStops: rightTabStop, after: 40 }),

      para([txt("Implemented data validation techniques ensuring accuracy and consistency, reducing discrepancies by 30%.")],
        { bullet: true }),
      para([txt("Extracted, transformed, and loaded (ETL) large datasets, improving reporting accuracy by 20%.")],
        { bullet: true }),
      para([txt("Developed and maintained Power BI dashboards to track data quality metrics and deliver actionable business insights.")],
        { bullet: true }),
      para([txt("Conducted statistical analysis and anomaly detection; collaborated with data governance stakeholders.")],
        { bullet: true }),

      para([], { after: 60 }),

      // ── PROJECTS ──────────────────────────────────────────────────────────
      sectionHead("Projects"),

      para([
        txt("Animal Image Classification using Transfer Learning", { bold: true, size: 21 }),
        txt("  |  "),
        link("GitHub", "https://github.com/Vijaybedage/ML-project"),
      ], { after: 40 }),
      para([txt("Built MobileNetV2-based deep learning model to classify 15 animal species from ~1,944 images. Achieved 93%+ validation accuracy with data augmentation and early stopping. Deployed interactive Streamlit web app for real-time prediction.")],
        { bullet: true }),
      para([txt("Tech: Python, TensorFlow, Keras, scikit-learn, Streamlit, Matplotlib")],
        { bullet: true }),

      para([], { after: 60 }),

      para([
        txt("Unicorn Startup Companies — Exploratory Data Analysis", { bold: true, size: 21 }),
        txt("  |  "),
        link("GitHub", "https://github.com/Vijaybedage/EDA-LIST-OF-UNICORN-STARTUP-COMPANIES-ANALYSIS"),
      ], { after: 40 }),
      para([txt("Analyzed global unicorn startups dataset to identify valuation trends, sector dominance, and geographic distribution using Python, Pandas, Matplotlib, and Seaborn.")],
        { bullet: true }),

      para([], { after: 60 }),

      para([
        txt("Summer Olympic Medals Analysis (1976–2008) — EDA", { bold: true, size: 21 }),
        txt("  |  "),
        link("GitHub", "https://github.com/Vijaybedage/Vijay"),
      ], { after: 40 }),
      para([txt("Performed in-depth EDA on Olympic medal data to uncover trends in country dominance, gender parity, and athletic diversity across 32 years.")],
        { bullet: true }),

      para([], { after: 60 }),

      // ── EDUCATION ─────────────────────────────────────────────────────────
      sectionHead("Education"),

      para([
        txt("Master of Computer Applications (MCA)  —  KLE Technological University, Hubli", { bold: true }),
        txt("\t"),
        txt("2025 – 2027 (Pursuing)", { italic: true, color: GRAY }),
      ], { tabStops: rightTabStop, after: 60 }),

      para([
        txt("Bachelor of Science (Physics, Computer Science)  —  Rani Channamma University, Belagavi", { bold: true }),
        txt("\t"),
        txt("2021 – 2024  |  65.40%", { italic: true, color: GRAY }),
      ], { tabStops: rightTabStop, after: 60 }),

      para([
        txt("Class XII (PUC Science)  —  R.D. PU Science College, Chikodi", { bold: true }),
        txt("\t"),
        txt("2019 – 2021  |  72.16%", { italic: true, color: GRAY }),
      ], { tabStops: rightTabStop, after: 60 }),

      para([
        txt("SSLC (Class X)  —  Karnataka Secondary Education Examination Board", { bold: true }),
        txt("\t"),
        txt("2019  |  77.44%", { italic: true, color: GRAY }),
      ], { tabStops: rightTabStop, after: 80 }),

      // ── CERTIFICATIONS ────────────────────────────────────────────────────
      sectionHead("Certifications"),

      para([txt("Machine Learning Internship Certificate  —  Unified Mentor Pvt Ltd (2025)")], { bullet: true }),
      para([txt("Data Analytics using Python  —  Leosias Technologies (2025)")], { bullet: true }),
      para([txt("Advanced MySQL  —  Leosias Technologies")], { bullet: true }),
      para([txt("Power BI  —  Leosias Technologies")], { bullet: true }),
    ]
  }]
});

// ── OUTPUT ────────────────────────────────────────────────────────────────────
const outputPath = path.join(__dirname, "VIJAY_RESUME_ATS.docx");
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(outputPath, buf);
  console.log(`Resume saved to: ${outputPath}`);
});
