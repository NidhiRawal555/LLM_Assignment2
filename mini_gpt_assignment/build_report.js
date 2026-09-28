const {
  Document, Packer, Paragraph, TextRun, HeadingLevel, Table, TableRow, TableCell,
  WidthType, ShadingType, BorderStyle, AlignmentType, LevelFormat
} = require("docx");
const fs = require("fs");

const PAGE_WIDTH = 12240, PAGE_HEIGHT = 15840; // US Letter, DXA

function h1(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_1, spacing: { before: 300, after: 150 } });
}
function h2(text) {
  return new Paragraph({ text, heading: HeadingLevel.HEADING_2, spacing: { before: 200, after: 100 } });
}
function body(text, opts = {}) {
  return new Paragraph({ children: [new TextRun({ text, ...opts })], spacing: { after: 120 } });
}
function placeholder(text) {
  return new Paragraph({
    children: [new TextRun({ text, italics: true, color: "808080" })],
    spacing: { after: 160 },
  });
}
function bullet(text) {
  return new Paragraph({ text, bullet: { level: 0 }, spacing: { after: 60 } });
}

function simpleTable(headerCells, rowsOfCells, colWidths) {
  const totalWidth = colWidths.reduce((a, b) => a + b, 0);
  const mkCell = (text, isHeader) => new TableCell({
    width: { size: colWidths[0], type: WidthType.DXA },
    shading: isHeader ? { type: ShadingType.CLEAR, fill: "D9E2F3" } : undefined,
    children: [new Paragraph({ children: [new TextRun({ text, bold: isHeader })] })],
  });
  const headerRow = new TableRow({
    children: headerCells.map((t, i) => new TableCell({
      width: { size: colWidths[i], type: WidthType.DXA },
      shading: { type: ShadingType.CLEAR, fill: "D9E2F3" },
      children: [new Paragraph({ children: [new TextRun({ text: t, bold: true })] })],
    })),
  });
  const bodyRows = rowsOfCells.map(cells => new TableRow({
    children: cells.map((t, i) => new TableCell({
      width: { size: colWidths[i], type: WidthType.DXA },
      children: [new Paragraph({ text: String(t) })],
    })),
  }));
  return new Table({
    width: { size: totalWidth, type: WidthType.DXA },
    columnWidths: colWidths,
    rows: [headerRow, ...bodyRows],
  });
}

const doc = new Document({
  sections: [{
    properties: { page: { size: { width: PAGE_WIDTH, height: PAGE_HEIGHT } } },
    children: [
      new Paragraph({
        children: [new TextRun({ text: "Building a Small-Scale Foundation Model from Scratch", bold: true, size: 32 })],
        spacing: { after: 60 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Assignment 2 — Report", size: 24, color: "555555" })],
        spacing: { after: 40 },
      }),
      new Paragraph({
        children: [new TextRun({ text: "Name: [Your Name]   |   Course: [Course Code]   |   Date: [Date]", size: 20, color: "808080" })],
        spacing: { after: 300 },
      }),

      h1("1. Model Architecture and Parameters"),
      body("Describe the mini-GPT you trained: number of transformer layers, embedding dimension, number of attention heads, positional encoding scheme, and total parameter count (printed automatically by model.py on startup). Briefly explain the role of each component (embeddings, causal self-attention, feed-forward block, layer normalization)."),
      placeholder("[Fill in: paste the parameter count printed by model.py, and your chosen n_layer / n_embd / n_head values.]"),
      simpleTable(
        ["Hyperparameter", "Value"],
        [
          ["n_layer (transformer blocks)", "[e.g., 2]"],
          ["n_embd (embedding dim)", "[e.g., 128]"],
          ["n_head (attention heads)", "[e.g., 4]"],
          ["block_size (sequence length)", "[e.g., 64]"],
          ["Total parameters", "[from model.py output]"],
        ],
        [4500, 4500]
      ),

      h1("2. Dataset Details"),
      body("Describe the dataset carried over from Assignment 1: source/domain of the text, size (characters/tokens), tokenization scheme (character-level, BPE, word-piece, etc.), vocabulary size, and the train/validation split used."),
      placeholder("[Fill in: dataset name/source, total token count, vocab_size, train/val split percentages.]"),

      h1("3. Training Setup and Hyperparameter Experiments"),
      body("Describe the training configuration: optimizer (AdamW), loss function (cross-entropy for next-token prediction), number of epochs, and hardware used. Then summarize the hyperparameter sweep run via sweep.py."),
      h2("3.1 Training configuration"),
      placeholder("[Fill in: batch size, learning rate, epochs, hardware (e.g., Colab T4 GPU / CPU), training wall-clock time.]"),
      h2("3.2 Hyperparameter experiment results"),
      body("Report the final validation loss and perplexity for each configuration tried (learning rate, batch size, number of layers, embedding size), e.g.:"),
      simpleTable(
        ["Experiment", "Changed Parameter", "Final Val Loss", "Final Val Perplexity"],
        [
          ["lr_1e-3", "learning_rate=1e-3", "[fill in]", "[fill in]"],
          ["lr_5e-4", "learning_rate=5e-4", "[fill in]", "[fill in]"],
          ["batch_16", "batch_size=16", "[fill in]", "[fill in]"],
          ["batch_64", "batch_size=64", "[fill in]", "[fill in]"],
          ["1_layer", "n_layer=1", "[fill in]", "[fill in]"],
          ["2_layer", "n_layer=2", "[fill in]", "[fill in]"],
          ["embd_64", "n_embd=64", "[fill in]", "[fill in]"],
          ["embd_256", "n_embd=256", "[fill in]", "[fill in]"],
        ],
        [2000, 2800, 2100, 2100]
      ),
      body(""),
      body("Insert training_curves.png (produced by visualize.py) below, showing loss and perplexity across epochs for the compared runs.", { bold: false }),
      placeholder("[Insert Figure 1: training_curves.png here]"),

      h1("4. Observations and Challenges"),
      body("Discuss what you learned from the experiments, for example:"),
      bullet("Which hyperparameter had the largest effect on validation loss/perplexity, and why you think that is."),
      bullet("Signs of overfitting or underfitting observed (e.g., training loss dropping while validation loss plateaus or rises)."),
      bullet("Any qualitative assessment of generated text samples (via generate.py), even if the model is too small/undertrained to produce fluent output."),
      bullet("Practical challenges encountered (e.g., tokenization mismatches with Assignment 1 data, memory/compute limits, tuning block_size vs. batch_size)."),
      placeholder("[Fill in your own observations here — this section should reflect your actual run, not a generic write-up.]"),

      h1("Appendix: Repository Structure"),
      body("config.py — hyperparameters  |  data.py — dataset loading & batching  |  model.py — MiniGPT architecture  |  train.py — training loop, logging, checkpointing  |  sweep.py — hyperparameter experiments  |  visualize.py — plotting  |  generate.py — text generation demo", { italics: true, size: 18 }),
    ],
  }],
});

Packer.toBuffer(doc).then(buf => {
  fs.writeFileSync("/home/claude/mini_gpt_assignment/report_template.docx", buf);
  console.log("done");
});
