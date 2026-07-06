const {
  Document, Packer, Paragraph, TextRun,
  AlignmentType, LevelFormat, BorderStyle,
  TabStopType, TabStopPosition, ExternalHyperlink, UnderlineType
} = require('docx');
const fs = require('fs');
const path = require('path');

// ── COLORS ──────────────────────────────────────────────────────────────────
const NAVY   = "1B2A4A";
const DARK   = "2C3E50";
const BODY   = "333333";
const SUBTLE = "555555";
const LINK_C = "2E75B6";
const LINE_C = "B0C4DE";

// ── HELPERS ─────────────────────────────────────────────────────────────────
const txt = (text, opts = {}) => new TextRun({
  text,
  font: "Calibri",
  size: opts.size || 21,
  bold: opts.bold || false,
  color: opts.color || BODY,
  italics: opts.italic || false,
  underline: opts.underline ? { type: UnderlineType.SINGLE } : undefined,
});

const link = (text, url) => new ExternalHyperlink({
  link: url,
  children: [new TextRun({
    text, font: "Calibri", size: 21, color: LINK_C,
    underline: { type: UnderlineType.SINGLE }
  })]
});

const para = (children, opts = {}) => new Paragraph({
  children: Array.isArray(children) ? children : [children],
  alignment: opts.align || AlignmentType.LEFT,
  spacing: { before: opts.before || 0, after: opts.after || 60, line: opts.line || 276 },
  tabStops: opts.tabStops || [],
  numbering: opts.bullet ? { reference: "bullets", level: 0 } : undefined,
  border: opts.borderBottom ? {
    bottom: { style: BorderStyle.SINGLE, size: 4, color: LINE_C, space: 4 }
  } : undefined,
});

const heading = (title) => para(
  [txt(title.toUpperCase(), { bold: true, size: 23, color: NAVY })],
  { before: 200, after: 60, borderBottom: true }
);

const rightTab = [{ type: TabStopType.RIGHT, position: TabStopPosition.MAX }];

// ── BULLET CONFIG ───────────────────────────────────────────────────────────
const numbering = {
  config: [{
    reference: "bullets",
    levels: [{
      level: 0, format: LevelFormat.BULLET, text: "\u2022",
      alignment: AlignmentType.LEFT,
      style: { paragraph: { indent: { left: 480, hanging: 240 }, spacing: { after: 50, line: 276 } } }
    }]
  }]
};

// ── DOCUMENT ────────────────────────────────────────────────────────────────
const doc = new Document({
  numbering,
  styles: {
    default: { document: { run: { font: "Calibri", size: 21 } } }
  },
  sections: [{
    properties: {
      page: {
        size: { width: 12240, height: 15840 },
        margin: { top: 720, right: 1080, bottom: 720, left: 1080 }
      }
    },
    children: [

      // ════════════════════════════════════════════════════════════════════════
      // NAME & CONTACT
      // ════════════════════════════════════════════════════════════════════════
      para([txt("Vijay Bedage", { bold: true, size: 40, color: NAVY })],
        { align: AlignmentType.CENTER, after: 10 }),

      para([txt("Data Analyst  ·  ML Practitioner", { size: 22, color: SUBTLE, italic: true })],
        { align: AlignmentType.CENTER, after: 80 }),

      para([
        txt("Phone: +91 7795745576", { size: 19, color: SUBTLE }),
        txt("  ·  ", { size: 19, color: SUBTLE }),
        link("Vijaybedage24@gmail.com", "mailto:Vijaybedage24@gmail.com"),
        txt("  ·  ", { size: 19, color: SUBTLE }),
        link("LinkedIn", "https://www.linkedin.com/in/vijay-bedage-226457315/"),
        txt("  ·  ", { size: 19, color: SUBTLE }),
        link("GitHub", "https://github.com/Vijaybedage"),
      ], { align: AlignmentType.CENTER, after: 100 }),

      para([
        txt("Belagavi, Karnataka, India", { size: 19, color: SUBTLE }),
      ], { align: AlignmentType.CENTER, after: 120 }),

      // ════════════════════════════════════════════════════════════════════════
      // PROFILE
      // ════════════════════════════════════════════════════════════════════════
      heading("Profile"),

      para([txt(
        "I'm a Computer Science graduate currently pursuing my MCA at KLE Technological University. " +
        "Over the past year I've worked across two internships — one focused on data analytics and the other on machine learning — " +
        "and what I enjoy most is taking messy, real-world data and making sense of it. " +
        "I'm comfortable writing Python scripts for analysis, building ML models, and putting together dashboards in Power BI. " +
        "I don't claim to know everything, but I pick things up fast and I'm genuinely curious about how data can solve practical problems."
      )], { after: 100 }),

      // ════════════════════════════════════════════════════════════════════════
      // SKILLS
      // ════════════════════════════════════════════════════════════════════════
      heading("Technical Skills"),

      para([
        txt("Programming: ", { bold: true }),
        txt("Python, R, C, C++, SQL"),
      ], { after: 50 }),
      para([
        txt("Data & ML Libraries: ", { bold: true }),
        txt("Pandas, NumPy, Matplotlib, Seaborn, scikit-learn, TensorFlow, Keras"),
      ], { after: 50 }),
      para([
        txt("BI & Visualization: ", { bold: true }),
        txt("Power BI (DAX, data modeling), Tableau, Advanced Excel (pivot tables, VLOOKUP, macros)"),
      ], { after: 50 }),
      para([
        txt("Databases: ", { bold: true }),
        txt("MySQL — comfortable writing complex joins, subqueries, and stored procedures"),
      ], { after: 50 }),
      para([
        txt("Tools & Platforms: ", { bold: true }),
        txt("Jupyter Notebook, VS Code, Google Colab, Git, GitHub, Streamlit"),
      ], { after: 50 }),
      para([
        txt("Core Concepts: ", { bold: true }),
        txt("Exploratory Data Analysis, Supervised & Unsupervised Learning, Transfer Learning, " +
            "Data Cleaning & Wrangling, ETL Pipelines, Statistical Testing, Data Visualization, SDLC"),
      ], { after: 100 }),

      // ════════════════════════════════════════════════════════════════════════
      // WORK EXPERIENCE
      // ════════════════════════════════════════════════════════════════════════
      heading("Work Experience"),

      // — ML Intern ——————————————————————————————————————————————————————————
      para([
        txt("Machine Learning Intern", { bold: true, size: 22, color: DARK }),
      ], { after: 10 }),
      para([
        txt("Unified Mentor Pvt Ltd", { italic: true, color: SUBTLE }),
        txt("\t"),
        txt("Jun 2025 – Sep 2025", { italic: true, color: SUBTLE }),
      ], { tabStops: rightTab, after: 50 }),

      para([txt(
        "My main project here was building an animal image classification system from scratch. " +
        "I collected around 1,944 images across 15 animal species, set up the data pipeline, and trained a " +
        "MobileNetV2-based model using transfer learning. The model reached 93%+ validation accuracy, " +
        "which honestly took a few rounds of tuning — I experimented with different learning rates, " +
        "augmentation strategies, and early stopping thresholds before I got there."
      )], { bullet: true }),

      para([txt(
        "I also deployed the final model as a Streamlit web app so anyone on the team could upload an image " +
        "and get a prediction instantly. It was a good exercise in going from notebook code to something usable."
      )], { bullet: true }),

      para([txt(
        "Along the way, I got comfortable with confusion matrices, classification reports, and debugging " +
        "issues like class imbalance and overfitting — things that textbooks mention but feel very different " +
        "when you're dealing with them on actual data."
      )], { bullet: true }),

      para([], { after: 80 }),

      // — Data Analytics Intern ——————————————————————————————————————————————
      para([
        txt("Data Analytics Intern", { bold: true, size: 22, color: DARK }),
      ], { after: 10 }),
      para([
        txt("Leosias Technologies", { italic: true, color: SUBTLE }),
        txt("\t"),
        txt("Oct 2024 – Mar 2025", { italic: true, color: SUBTLE }),
      ], { tabStops: rightTab, after: 50 }),

      para([txt(
        "I spent most of my time here cleaning and validating datasets — the kind of work that isn't glamorous " +
        "but matters a lot. I wrote validation checks that caught inconsistencies early, and over the six months " +
        "we managed to cut data discrepancies by about 30%."
      )], { bullet: true }),

      para([txt(
        "Built and maintained Power BI dashboards that the operations team actually used day-to-day. " +
        "One dashboard tracked data quality metrics across sources; another gave a weekly snapshot of reporting accuracy. " +
        "Reporting accuracy improved by roughly 20% after we automated the ETL pipeline I helped put together."
      )], { bullet: true }),

      para([txt(
        "Ran basic statistical analyses — mostly hypothesis testing and anomaly detection — to flag unusual patterns " +
        "in client data. Worked closely with the data governance team to document data lineage and set up quality standards."
      )], { bullet: true }),

      para([txt(
        "This internship taught me that good analysis starts long before you open a notebook. " +
        "If the data is bad, nothing downstream will save you."
      )], { bullet: true }),

      para([], { after: 80 }),

      // ════════════════════════════════════════════════════════════════════════
      // PROJECTS
      // ════════════════════════════════════════════════════════════════════════
      heading("Projects"),

      // — Project 1 ——————————————————————————————————————————————————————————
      para([
        txt("Animal Image Classification using Transfer Learning", { bold: true, size: 22, color: DARK }),
        txt("   "),
        link("View on GitHub", "https://github.com/Vijaybedage/ML-project"),
      ], { after: 40 }),

      para([txt(
        "This was my main internship project. I used TensorFlow and Keras to fine-tune a pre-trained MobileNetV2 model " +
        "on a custom dataset of 15 animal classes (~1,944 images total). I handled everything from data collection and " +
        "augmentation (random flips, rotations, zoom) to training and evaluation. The model hit 93%+ validation accuracy. " +
        "I then wrapped it in a Streamlit app for real-time predictions."
      )], { bullet: true }),
      para([txt(
        "Stack: Python, TensorFlow, Keras, scikit-learn, Streamlit, Matplotlib, NumPy"
      )], { bullet: true }),

      para([], { after: 60 }),

      // — Project 2 ——————————————————————————————————————————————————————————
      para([
        txt("Unicorn Startup Companies — Exploratory Data Analysis", { bold: true, size: 22, color: DARK }),
        txt("   "),
        link("View on GitHub", "https://github.com/Vijaybedage/EDA-LIST-OF-UNICORN-STARTUP-COMPANIES-ANALYSIS"),
      ], { after: 40 }),

      para([txt(
        "Pulled apart a dataset of global unicorn startups to find patterns in valuations, sectors, and geography. " +
        "Most of the interesting findings came from cross-referencing founding year with sector — fintech unicorns, " +
        "for instance, exploded after 2018. Built all visualizations in Matplotlib and Seaborn."
      )], { bullet: true }),
      para([txt(
        "Stack: Python, Pandas, Matplotlib, Seaborn, Jupyter Notebook"
      )], { bullet: true }),

      para([], { after: 60 }),

      // — Project 3 ——————————————————————————————————————————————————————————
      para([
        txt("Summer Olympic Medals Analysis (1976–2008)", { bold: true, size: 22, color: DARK }),
        txt("   "),
        link("View on GitHub", "https://github.com/Vijaybedage/Vijay"),
      ], { after: 40 }),

      para([txt(
        "Analyzed 32 years of Olympic medal data looking at how country dominance shifted over time, " +
        "whether gender representation improved (it did, but slowly), and which sports saw the most competitive diversity. " +
        "The dataset had some messy country-name issues due to political changes, which made cleaning it a good learning exercise."
      )], { bullet: true }),
      para([txt(
        "Stack: Python, Pandas, Matplotlib, Seaborn"
      )], { bullet: true }),

      para([], { after: 80 }),

      // ════════════════════════════════════════════════════════════════════════
      // EDUCATION
      // ════════════════════════════════════════════════════════════════════════
      heading("Education"),

      para([
        txt("Master of Computer Applications (MCA)", { bold: true, color: DARK }),
      ], { after: 10 }),
      para([
        txt("KLE Technological University, Hubli", { italic: true, color: SUBTLE }),
        txt("\t"),
        txt("2025 – 2027 (currently pursuing)", { italic: true, color: SUBTLE }),
      ], { tabStops: rightTab, after: 60 }),

      para([
        txt("Bachelor of Science — Physics & Computer Science", { bold: true, color: DARK }),
      ], { after: 10 }),
      para([
        txt("Rani Channamma University, Belagavi", { italic: true, color: SUBTLE }),
        txt("\t"),
        txt("2021 – 2024  |  65.40%", { italic: true, color: SUBTLE }),
      ], { tabStops: rightTab, after: 60 }),

      para([
        txt("Class XII — PUC Science", { bold: true, color: DARK }),
      ], { after: 10 }),
      para([
        txt("R.D. PU Science College, Chikodi", { italic: true, color: SUBTLE }),
        txt("\t"),
        txt("2019 – 2021  |  72.16%", { italic: true, color: SUBTLE }),
      ], { tabStops: rightTab, after: 60 }),

      para([
        txt("SSLC (Class X)", { bold: true, color: DARK }),
      ], { after: 10 }),
      para([
        txt("Karnataka Secondary Education Examination Board", { italic: true, color: SUBTLE }),
        txt("\t"),
        txt("2019  |  77.44%", { italic: true, color: SUBTLE }),
      ], { tabStops: rightTab, after: 100 }),

      // ════════════════════════════════════════════════════════════════════════
      // CERTIFICATIONS
      // ════════════════════════════════════════════════════════════════════════
      heading("Certifications"),

      para([txt("Machine Learning Internship Certificate — Unified Mentor Pvt Ltd, 2025")], { bullet: true }),
      para([txt("Data Analytics using Python — Leosias Technologies, 2025")], { bullet: true }),
      para([txt("Advanced MySQL — Leosias Technologies")], { bullet: true }),
      para([txt("Power BI — Leosias Technologies")], { bullet: true, after: 100 }),

      // ════════════════════════════════════════════════════════════════════════
      // LANGUAGES & INTERESTS
      // ════════════════════════════════════════════════════════════════════════
      heading("Languages"),

      para([txt("English (professional proficiency), Hindi (fluent), Kannada (native)")], { after: 100 }),

      heading("Interests"),

      para([txt(
        "I like working on side projects involving data — recently I've been exploring how different " +
        "image augmentation techniques affect model accuracy. Outside of tech, I follow cricket and enjoy reading " +
        "about how companies use analytics in decision-making."
      )], { after: 60 }),

    ]
  }]
});

// ── OUTPUT ───────────────────────────────────────────────────────────────────
const outputPath = path.join(__dirname, "VIJAY_CV.docx");
Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync(outputPath, buf);
  console.log(`CV saved to: ${outputPath}`);
});
