import * as pdfjsLib from 'pdfjs-dist';
import pdfWorkerUrl from 'pdfjs-dist/build/pdf.worker.min.mjs?url';
import { createWorker } from 'tesseract.js';

pdfjsLib.GlobalWorkerOptions.workerSrc = pdfWorkerUrl;

export type Chunk = {
  id: number;
  text: string;
  embedding: number[];
};

export type ProcessedDocument = {
  name: string;
  size: string;
  text: string;
  chunks: Chunk[];
  vocab: Map<string, number>;
  idf: Float64Array;
};

const CHUNK_SIZE = 250;
const CHUNK_OVERLAP = 50;
const STOPWORDS = new Set([
  'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been',
  'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
  'should', 'may', 'might', 'must', 'can', 'this', 'that', 'these', 'those', 'i',
  'you', 'he', 'she', 'it', 'we', 'they', 'what', 'which', 'who', 'when', 'where',
  'why', 'how', 'all', 'each', 'every', 'both', 'few', 'more', 'most', 'other',
  'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so', 'than', 'too',
  'very', 'just', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'from',
  'about', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
  'below', 'up', 'down', 'out', 'if', 'then', 'there', 'here', 'my', 'your',
]);

function tokenize(text: string): string[] {
  return text
    .toLowerCase()
    .split(/[^a-z0-9]+/)
    .filter((word) => word.length > 1 && !STOPWORDS.has(word));
}

function buildVocabulary(chunks: string[]): Map<string, number> {
  const vocab = new Map<string, number>();
  for (const chunk of chunks) {
    for (const token of tokenize(chunk)) {
      if (!vocab.has(token)) vocab.set(token, vocab.size);
    }
  }
  return vocab;
}

function tfidfVector(text: string, vocab: Map<string, number>, idf: Float64Array): number[] {
  const tokens = tokenize(text);
  const tf = new Map<number, number>();
  for (const token of tokens) {
    const index = vocab.get(token);
    if (index === undefined) continue;
    tf.set(index, (tf.get(index) ?? 0) + 1);
  }
  const vector = new Array(vocab.size).fill(0);
  const totalTokens = tokens.length || 1;
  for (const [index, count] of tf) {
    vector[index] = (count / totalTokens) * idf[index];
  }
  return vector;
}

function computeIdf(chunks: string[], vocab: Map<string, number>): Float64Array {
  const docCount = chunks.length || 1;
  const docFreq = new Float64Array(vocab.size);
  for (const chunk of chunks) {
    const tokens = new Set(tokenize(chunk));
    for (const token of tokens) {
      const index = vocab.get(token);
      if (index !== undefined) docFreq[index]++;
    }
  }
  const idf = new Float64Array(vocab.size);
  for (let i = 0; i < vocab.size; i++) {
    idf[i] = Math.log((1 + docCount) / (1 + docFreq[i])) + 1;
  }
  return idf;
}

function cosineSimilarity(a: number[], b: number[]): number {
  let dot = 0;
  let normA = 0;
  let normB = 0;
  const len = Math.min(a.length, b.length);
  for (let i = 0; i < len; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  if (normA === 0 || normB === 0) return 0;
  return dot / (Math.sqrt(normA) * Math.sqrt(normB));
}

function splitIntoChunks(text: string): string[] {
  const cleanText = text.replace(/\s+/g, ' ').trim();
  if (!cleanText) return [];
  const words = cleanText.split(' ');
  const chunks: string[] = [];
  let start = 0;
  while (start < words.length) {
    const end = Math.min(start + CHUNK_SIZE, words.length);
    chunks.push(words.slice(start, end).join(' '));
    if (end >= words.length) break;
    start += CHUNK_SIZE - CHUNK_OVERLAP;
  }
  return chunks;
}

function createEmbeddings(chunkTexts: string[]): { chunks: Chunk[]; vocab: Map<string, number>; idf: Float64Array } {
  const vocab = buildVocabulary(chunkTexts);
  const idf = computeIdf(chunkTexts, vocab);
  const chunks: Chunk[] = chunkTexts.map((text, index) => ({
    id: index,
    text,
    embedding: tfidfVector(text, vocab, idf),
  }));
  return { chunks, vocab, idf };
}

export function embedQuery(query: string, vocab: Map<string, number>, idf: Float64Array): number[] {
  return tfidfVector(query, vocab, idf);
}

export function similaritySearch(
  query: string,
  chunks: Chunk[],
  vocab: Map<string, number>,
  idf: Float64Array,
  topK = 3,
  threshold = 0.05,
): Chunk[] {
  if (chunks.length === 0) return [];
  const queryEmbedding = embedQuery(query, vocab, idf);
  const scored = chunks.map((chunk) => ({
    chunk,
    score: cosineSimilarity(queryEmbedding, chunk.embedding),
  }));
  scored.sort((a, b) => b.score - a.score);
  const results = scored.filter((s) => s.score > threshold).slice(0, topK);
  console.log('[RAG] Similarity search results:', results.map((s) => ({ score: s.score.toFixed(4), preview: s.chunk.text.slice(0, 80) })));
  return results.map((s) => s.chunk);
}

async function extractTextFromPdfPage(page: any): Promise<string> {
  const textContent = await page.getTextContent();
  return textContent.items
    .map((item: any) => ('str' in item ? item.str : ''))
    .join(' ')
    .replace(/\s+/g, ' ')
    .trim();
}

async function renderPageToCanvas(page: any): Promise<ImageData> {
  const viewport = page.getViewport({ scale: 2 });
  const canvas = document.createElement('canvas');
  canvas.width = viewport.width;
  canvas.height = viewport.height;
  const context = canvas.getContext('2d');
  if (!context) throw new Error('Canvas 2D context not available');
  await page.render({ canvasContext: context, viewport }).promise;
  return context.getImageData(0, 0, canvas.width, canvas.height);
}

async function ocrPage(page: any): Promise<string> {
  let worker: any;
  try {
    worker = await createWorker('eng', 1, {
      logger: () => {},
    });
    const imageData = await renderPageToCanvas(page);
    const { data } = await worker.recognize(imageData);
    return (data?.text ?? '').replace(/\s+/g, ' ').trim();
  } finally {
    if (worker) await worker.terminate();
  }
}

export async function processPdf(file: File): Promise<ProcessedDocument> {
  const arrayBuffer = await file.arrayBuffer();
  const pdf = await pdfjsLib.getDocument({ data: arrayBuffer }).promise;
  const numPages = pdf.numPages;
  const pageTexts: string[] = [];

  for (let pageNum = 1; pageNum <= numPages; pageNum++) {
    const page = await pdf.getPage(pageNum);
    let text = await extractTextFromPdfPage(page);
    if (text.length < 50) {
      console.log(`[RAG] Page ${pageNum}: text too short (${text.length} chars), running OCR...`);
      try {
        const ocrText = await ocrPage(page);
        if (ocrText.length > text.length) {
          text = ocrText;
        }
      } catch (err) {
        console.warn(`[RAG] OCR failed on page ${pageNum}:`, err);
      }
    }
    pageTexts.push(text);
  }

  const fullText = pageTexts.join('\n\n');
  console.log('[RAG] Extracted text length:', fullText.length, 'chars');

  const chunkTexts = splitIntoChunks(fullText);
  console.log('[RAG] Number of chunks:', chunkTexts.length);

  const { chunks, vocab, idf } = createEmbeddings(chunkTexts);
  console.log('[RAG] Number of embeddings:', chunks.length, '| Vocabulary size:', vocab.size);

  return {
    name: file.name,
    size: `${(file.size / 1024).toFixed(1)} KB`,
    text: fullText,
    chunks,
    vocab,
    idf,
  };
}

export function processTextFile(file: File, text: string): ProcessedDocument {
  const cleanText = text.replace(/[\u0000-\u0008\u000B\u000C\u000E-\u001F]/g, ' ').replace(/\s{2,}/g, ' ').trim();
  console.log('[RAG] Extracted text length:', cleanText.length, 'chars');

  const chunkTexts = splitIntoChunks(cleanText);
  console.log('[RAG] Number of chunks:', chunkTexts.length);

  const { chunks, vocab, idf } = createEmbeddings(chunkTexts);
  console.log('[RAG] Number of embeddings:', chunks.length, '| Vocabulary size:', vocab.size);

  return {
    name: file.name,
    size: `${(file.size / 1024).toFixed(1)} KB`,
    text: cleanText,
    chunks,
    vocab,
    idf,
  };
}
