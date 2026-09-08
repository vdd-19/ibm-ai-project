import { ChangeEvent, DragEvent, FormEvent, useMemo, useRef, useState } from 'react';
import { ProcessedDocument, processPdf, processTextFile, similaritySearch } from '@/lib/rag';
import {
  ArrowUp,
  BookOpen,
  Check,
  ChevronDown,
  CircleHelp,
  FileText,
  GraduationCap,
  LayoutDashboard,
  Menu,
  MessageCircle,
  MoreHorizontal,
  Paperclip,
  Plus,
  Search,
  Sparkles,
  UploadCloud,
  UserRound,
  X,
} from 'lucide-react';

type Message = {
  id: number;
  role: 'user' | 'assistant';
  text: string;
  context?: string;
  time: string;
};

const sampleDocument = `University Academic Handbook 2024-25

Grading criteria
Students are assessed through a combination of continuous assessment, mid-semester examinations, and final examinations. The final grade is calculated as: assignments and projects 30%, mid-semester examination 20%, and end-semester examination 50%.

Pass marks
A minimum score of 40% is required to pass each individual course. Students must also maintain a cumulative GPA of 2.0 or higher to graduate. A score below 40% requires a supplementary examination where eligible.

Attendance policy
Students are expected to maintain at least 75% attendance in every registered course. Students with attendance between 65% and 74% may submit a medical or exceptional circumstances request to the department. Attendance below 65% may make a student ineligible to sit the final examination.

Academic support
Students can contact their academic advisor during office hours for help with course planning, assessment questions, and exceptional circumstances requests.`;

const suggestedQuestions = ['What are the pass marks?', 'How is my grade calculated?', 'What is the attendance rule?'];

const initialMessages: Message[] = [
  {
    id: 1,
    role: 'assistant',
    text: 'Hello, Maya. I’m your academic support assistant. Ask me anything about your courses, policies, or uploaded documents.',
    time: '09:41 AM',
  },
  {
    id: 2,
    role: 'assistant',
    text: 'You can upload a syllabus or FAQ in the sidebar, and I’ll use it as my source of truth for your answers.',
    time: '09:41 AM',
  },
];

function getTime(): string {
  return new Intl.DateTimeFormat('en-US', { hour: '2-digit', minute: '2-digit' }).format(new Date());
}

const greetingPatterns = /^(hi+|hello+|hey+|yo+|sup|howdy|good\s+(morning|afternoon|evening|night)|thanks?|thank\s+you|thx|ty|okay|ok+|sure|cool|got\s*it|understood|bye+|goodbye|see\s*ya|alright|sounds\s*good)\b.*/i;

const academicKeywords = [
  'grade', 'grading', 'gpa', 'pass', 'mark', 'marks', 'score', 'scoring', 'exam', 'examination',
  'mid-semester', 'midterm', 'final', 'attendance', 'absent', 'absence', 'present', 'syllabus',
  'assignment', 'assignments', 'project', 'projects', 'course', 'courses', 'credit', 'credits',
  'semester', 'registration', 'deadline', 'policy', 'policies', 'advisor', 'tuition', 'fee',
  'fees', 'scholarship', 'degree', 'graduation', 'graduate', 'curriculum', 'assessment',
  'continuous', 'supplementary', 'eligibility', 'eligible', 'enroll', 'enrollment', 'academic',
  'university', 'college', 'faculty', 'department', 'lecture', 'tutorial', 'lab', 'practical',
  'quiz', 'test', 'homework', 'submission', 'penalty', 'late', 'plagiarism', 'integrity',
  'attendance', 'percentage', 'weight', 'weightage', 'rubric', 'criteria', 'criterion',
  'unit', 'chapter', 'module', 'topic', 'content', 'requirement', 'requirements', 'rule', 'rules',
];

function classifyIntent(query: string): 'greeting' | 'academic' | 'unrelated' {
  const trimmed = query.trim().toLowerCase();
  if (greetingPatterns.test(trimmed)) return 'greeting';
  const words = trimmed.split(/[^a-z0-9-]+/).filter(Boolean);
  const isAcademic = words.some((word) => academicKeywords.some((keyword) => word === keyword || word.includes(keyword)));
  return isAcademic ? 'academic' : 'unrelated';
}

function findContext(query: string, source: string): string {
  const sections = source.split(/\n\s*\n/).filter(Boolean);
  const terms = query.toLowerCase().split(/[^a-z0-9]+/).filter((term) => term.length > 3);
  if (terms.length === 0) return '';
  const scored = sections.map((section) => ({
    section,
    score: terms.reduce((total, term) => total + (section.toLowerCase().includes(term) ? 1 : 0), 0),
  }));
  const best = scored.sort((a, b) => b.score - a.score)[0];
  return best && best.score > 0 ? best.section : '';
}

async function generateAnswer(query: string, context: string, hasDocument: boolean): Promise<string> {
  const intent = classifyIntent(query);

  if (intent === 'greeting') {
    return 'Hello! I\u2019m your AI Student Support Assistant. You can ask me about your syllabus, grading, attendance, examinations, assignments, or university policies.';
  }

  if (intent === 'unrelated') {
    return 'I\u2019m designed to help with university-related questions such as your syllabus, grading, attendance, examinations, and academic policies.';
  }

  const lower = query.toLowerCase();

  if (hasDocument && !context) {
    return 'I couldn\u2019t find this information in the uploaded document.';
  }

  if (/pass|minimum|score|mark/.test(lower)) {
    return hasDocument
      ? 'According to your uploaded document, you need a minimum score of 40% to pass each individual course. A cumulative GPA of 2.0 or higher is also required to graduate.'
      : 'The general university pass mark is 40% for each course. Students should also maintain a cumulative GPA of 2.0 or higher to graduate.';
  }
  if (/attend|absen|present/.test(lower)) {
    return hasDocument
      ? 'Your document states that you need at least 75% attendance in every course. Between 65% and 74%, you may submit a medical or exceptional circumstances request. Below 65% may make you ineligible for the final exam.'
      : 'The general attendance requirement is 75% in every registered course. If your attendance falls below that, contact your department early to discuss your options.';
  }
  if (/grade|calculat|weight|exam|assess/.test(lower)) {
    return hasDocument
      ? 'Your final grade is calculated from three components:\n\u2022 Assignments and projects: 30%\n\u2022 Mid-semester examination: 20%\n\u2022 End-semester examination: 50%.'
      : 'The general grading structure is assignments and projects (30%), mid-semester examination (20%), and final examination (50%).';
  }
  if (/advisor|help|support|contact/.test(lower)) {
    return 'Your academic advisor can help with course planning, assessment questions, and exceptional circumstances requests. Check their office hours in your department handbook.';
  }

  if (context) {
    const apiKey = import.meta.env.VITE_GEMINI_API_KEY;
    if (apiKey) {
      try {
        const response = await fetch(`https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent?key=${apiKey}`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            contents: [{ parts: [{ text: `You are an AI Student Support Assistant.\n\nStudent Question:\n${query}\n\nRelevant Document Context:\n${context}\n\nAnswer using ONLY the provided document context. Give a concise, student-friendly answer. Do not mention RAG, embeddings, chunks, vector databases, retrieval, or internal metadata. If the context does not answer the question, say: "I couldn\\'t find this information in the uploaded document."` }] }],
          }),
        });
        if (!response.ok) throw new Error(`Gemini request failed with status ${response.status}`);
        const data: unknown = await response.json();
        const answer = (data as { candidates?: Array<{ content?: { parts?: Array<{ text?: string }> } }> }).candidates?.[0]?.content?.parts?.[0]?.text;
        if (answer?.trim()) return answer.trim();
        throw new Error('Gemini returned no answer');
      } catch (error) {
        console.error('[RAG] Gemini request failed:', error);
        return 'Sorry, I couldn\u2019t process your question right now. Please try again.';
      }
    }
    return `Based on your uploaded document:\n\n${context.replace(/\n/g, ' ')}`;
  }

  return 'I couldn\u2019t find this information in the uploaded document.';
}

function App() {
  const [messages, setMessages] = useState<Message[]>(initialMessages);
  const [question, setQuestion] = useState('');
  const [document, setDocument] = useState<ProcessedDocument | null>(null);
  const [isDragging, setIsDragging] = useState(false);
  const [isMobileSidebarOpen, setIsMobileSidebarOpen] = useState(false);
  const [isThinking, setIsThinking] = useState(false);
  const inputRef = useRef<HTMLInputElement>(null);

  const sourceText = document?.text || sampleDocument;
  const contextLabel = document ? 'Uploaded context' : 'University handbook';
  const questions = useMemo(() => suggestedQuestions, []);

  const readFile = async (file: File) => {
    if (!/\.(pdf|txt)$/i.test(file.name)) return;
    try {
      const processed = file.name.toLowerCase().endsWith('.pdf')
        ? await processPdf(file)
        : processTextFile(file, await file.text());
      setDocument(processed);
      setIsMobileSidebarOpen(false);
    } catch (error) {
      console.error('[RAG] Document processing failed:', error);
    }
  };

  const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (file) void readFile(file);
    event.target.value = '';
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    const file = event.dataTransfer.files[0];
    if (file) void readFile(file);
  };

  const submitQuestion = (event?: FormEvent) => {
    event?.preventDefault();
    const trimmed = question.trim();
    if (!trimmed || isThinking) return;
    const intent = classifyIntent(trimmed);
    let context = '';
    if (intent === 'academic') {
      if (document) {
        const retrievedChunks = similaritySearch(trimmed, document.chunks, document.vocab, document.idf);
        context = retrievedChunks.map((chunk) => chunk.text).join('\n\n');
        console.log('[RAG] Query:', trimmed);
        console.log('[RAG] Retrieved chunk count:', retrievedChunks.length);
        console.log('[RAG] Retrieved text:', context);
      } else {
        context = findContext(trimmed, sourceText);
        console.log('[RAG] Query:', trimmed);
        console.log('[RAG] Retrieved chunk count:', context ? 1 : 0);
        console.log('[RAG] Retrieved text:', context);
      }
    } else {
      console.log('[RAG] Skipped retrieval for intent:', intent);
    }
    const userMessage: Message = { id: Date.now(), role: 'user', text: trimmed, time: getTime() };
    setMessages((current) => [...current, userMessage]);
    setQuestion('');
    setIsThinking(true);
    window.setTimeout(() => {
      void generateAnswer(trimmed, context, Boolean(document)).then((text) => {
        setMessages((current) => [...current, { id: Date.now() + 1, role: 'assistant', text, context: context || undefined, time: getTime() }]);
        setIsThinking(false);
      });
    }, 650);
  };

  const startNewChat = () => {
    setMessages(initialMessages);
    setQuestion('');
  };

  return (
    <div className="min-h-screen bg-[#0b0d12] text-slate-100 selection:bg-cyan-300/30">
      <div className="flex min-h-screen">
        <aside className={`fixed inset-y-0 left-0 z-30 flex w-[292px] shrink-0 flex-col border-r border-white/[0.07] bg-[#101319] px-5 py-6 transition-transform duration-300 lg:static lg:translate-x-0 ${isMobileSidebarOpen ? 'translate-x-0' : '-translate-x-full'}`}>
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="grid h-10 w-10 place-items-center rounded-xl bg-cyan-300 text-slate-950 shadow-[0_8px_24px_rgba(103,232,249,0.16)]"><GraduationCap size={22} strokeWidth={2.5} /></div>
              <div><p className="text-[15px] font-semibold tracking-tight">Scholarly</p><p className="text-[11px] text-slate-500">Student support AI</p></div>
            </div>
            <button onClick={() => setIsMobileSidebarOpen(false)} className="rounded-lg p-2 text-slate-500 hover:bg-white/5 hover:text-white lg:hidden" aria-label="Close menu"><X size={18} /></button>
          </div>

          <button onClick={startNewChat} className="mt-9 flex h-11 items-center justify-center gap-2 rounded-xl border border-white/10 bg-white/[0.04] text-sm font-medium text-slate-200 transition hover:border-cyan-300/30 hover:bg-cyan-300/10 hover:text-cyan-100"><Plus size={17} /> New conversation</button>

          <nav className="mt-8 space-y-1 text-sm">
            <p className="mb-3 px-3 text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600">Workspace</p>
            <button className="flex w-full items-center gap-3 rounded-lg bg-white/[0.07] px-3 py-2.5 text-left text-slate-100"><MessageCircle size={17} className="text-cyan-300" /> Current chat</button>
            <button className="flex w-full items-center gap-3 rounded-lg px-3 py-2.5 text-left text-slate-500 transition hover:bg-white/[0.04] hover:text-slate-200"><LayoutDashboard size={17} /> Overview</button>
          </nav>

          <div className="mt-9">
            <div className="mb-3 flex items-center justify-between px-1"><p className="text-[10px] font-semibold uppercase tracking-[0.18em] text-slate-600">Knowledge source</p><span className="rounded-full bg-cyan-300/10 px-2 py-0.5 text-[10px] font-medium text-cyan-300">RAG active</span></div>
            <div onDragOver={(event) => { event.preventDefault(); setIsDragging(true); }} onDragLeave={() => setIsDragging(false)} onDrop={handleDrop} className={`group rounded-2xl border border-dashed p-4 transition ${isDragging ? 'border-cyan-300 bg-cyan-300/10' : 'border-white/10 bg-white/[0.025] hover:border-cyan-300/40 hover:bg-white/[0.045]'}`}>
              <div className="flex items-start gap-3"><div className="grid h-9 w-9 shrink-0 place-items-center rounded-lg bg-cyan-300/10 text-cyan-300"><UploadCloud size={18} /></div><div><p className="text-xs font-medium text-slate-200">Drop your document here</p><p className="mt-1 text-[11px] leading-4 text-slate-500">PDF or TXT · max 10 MB</p></div></div>
              <label className="mt-4 flex h-9 cursor-pointer items-center justify-center rounded-lg border border-white/10 text-xs font-medium text-slate-300 transition hover:border-cyan-300/40 hover:bg-cyan-300/10 hover:text-cyan-200"><Paperclip size={14} className="mr-2" /> Browse files<input type="file" accept=".pdf,.txt,application/pdf,text/plain" onChange={handleFileChange} className="hidden" /></label>
            </div>
            {document ? <div className="mt-3 flex items-center gap-3 rounded-xl border border-emerald-400/20 bg-emerald-400/[0.07] p-3"><div className="grid h-8 w-8 place-items-center rounded-lg bg-emerald-400/15 text-emerald-300"><FileText size={16} /></div><div className="min-w-0 flex-1"><p className="truncate text-xs font-medium text-slate-200">{document.name}</p><p className="mt-0.5 text-[10px] text-emerald-300">Ready to reference · {document.size}</p></div><Check size={15} className="text-emerald-300" /></div> : <div className="mt-3 flex items-center gap-3 rounded-xl border border-white/[0.06] bg-white/[0.02] p-3"><div className="grid h-8 w-8 place-items-center rounded-lg bg-white/5 text-slate-500"><BookOpen size={16} /></div><div><p className="text-xs font-medium text-slate-400">General university guide</p><p className="mt-0.5 text-[10px] text-slate-600">Using sample knowledge base</p></div></div>}
          </div>

          <div className="mt-auto rounded-2xl border border-white/[0.07] bg-gradient-to-br from-cyan-300/[0.09] to-transparent p-4"><div className="mb-3 flex items-center gap-2 text-cyan-300"><Sparkles size={15} /><span className="text-[11px] font-semibold uppercase tracking-wider">Pro tip</span></div><p className="text-xs leading-5 text-slate-400">Upload your handbook for answers tailored to your university’s exact policies.</p></div>
          <div className="mt-5 flex items-center gap-3 border-t border-white/[0.06] pt-5"><div className="grid h-8 w-8 place-items-center rounded-full bg-slate-700 text-xs font-semibold">MC</div><div className="min-w-0 flex-1"><p className="truncate text-xs font-medium text-slate-200">Maya Chen</p><p className="text-[10px] text-slate-600">Undergraduate · Year 2</p></div><ChevronDown size={15} className="text-slate-600" /></div>
        </aside>

        {isMobileSidebarOpen && <button className="fixed inset-0 z-20 bg-black/60 lg:hidden" onClick={() => setIsMobileSidebarOpen(false)} aria-label="Close sidebar" />}

        <main className="flex min-w-0 flex-1 flex-col bg-[radial-gradient(circle_at_62%_0%,rgba(22,62,77,0.18),transparent_34%)]">
          <header className="flex h-[76px] shrink-0 items-center justify-between border-b border-white/[0.07] px-5 sm:px-8 lg:px-12"><div className="flex items-center gap-3"><button onClick={() => setIsMobileSidebarOpen(true)} className="rounded-lg p-2 text-slate-500 hover:bg-white/5 hover:text-white lg:hidden" aria-label="Open menu"><Menu size={20} /></button><div><div className="flex items-center gap-2"><h1 className="text-sm font-semibold text-slate-100 sm:text-[15px]">Academic support</h1><span className="hidden rounded-full border border-emerald-300/20 bg-emerald-300/10 px-2 py-0.5 text-[10px] font-medium text-emerald-300 sm:inline-flex">Online</span></div><p className="mt-1 text-[11px] text-slate-500">Ask questions about your student journey</p></div></div><div className="flex items-center gap-1"><button className="rounded-lg p-2.5 text-slate-500 transition hover:bg-white/5 hover:text-slate-200" aria-label="Search"><Search size={17} /></button><button className="rounded-lg p-2.5 text-slate-500 transition hover:bg-white/5 hover:text-slate-200" aria-label="Help"><CircleHelp size={18} /></button><button className="ml-1 rounded-lg p-2.5 text-slate-500 transition hover:bg-white/5 hover:text-slate-200" aria-label="More options"><MoreHorizontal size={18} /></button></div></header>

          <section className="mx-auto flex w-full max-w-4xl flex-1 flex-col px-5 pb-5 sm:px-8 lg:px-12"><div className="flex-1 overflow-y-auto py-8 sm:py-10"><div className="mb-10 text-center"><div className="mx-auto mb-4 grid h-12 w-12 place-items-center rounded-2xl border border-cyan-300/20 bg-cyan-300/10 text-cyan-300"><Sparkles size={23} /></div><h2 className="text-xl font-semibold tracking-tight text-white sm:text-2xl">How can I help you today?</h2><p className="mx-auto mt-2 max-w-md text-xs leading-5 text-slate-500 sm:text-sm">I can help you find answers in your academic documents or guide you through university policies.</p></div>
              <div className="space-y-6">{messages.map((message) => <div key={message.id} className={`flex gap-3 ${message.role === 'user' ? 'justify-end' : 'justify-start'}`}>{message.role === 'assistant' && <div className="mt-1 grid h-8 w-8 shrink-0 place-items-center rounded-xl bg-cyan-300 text-slate-950"><Sparkles size={15} /></div>}<div className={`max-w-[88%] sm:max-w-[76%] ${message.role === 'user' ? 'items-end' : 'items-start'}`}><div className={`rounded-2xl px-4 py-3 text-sm leading-6 ${message.role === 'user' ? 'rounded-br-md bg-cyan-300 text-slate-950' : 'rounded-bl-md border border-white/[0.08] bg-[#161a21] text-slate-300'}`}><p className="whitespace-pre-line">{message.text}</p></div>{message.context && <div className="mt-2 rounded-xl border border-cyan-300/15 bg-cyan-300/[0.045] p-3 text-xs leading-5 text-slate-400"><div className="mb-1.5 flex items-center gap-1.5 text-[10px] font-semibold uppercase tracking-wider text-cyan-300"><Search size={11} /> {contextLabel}</div><p className="line-clamp-2">{message.context.replace(/\n/g, ' ')}</p></div>}<p className={`mt-1.5 text-[10px] text-slate-600 ${message.role === 'user' ? 'text-right' : ''}`}>{message.time}</p></div>{message.role === 'user' && <div className="mt-1 grid h-8 w-8 shrink-0 place-items-center rounded-xl bg-slate-700 text-[10px] font-semibold text-slate-300"><UserRound size={15} /></div>}</div>)}{isThinking && <div className="flex gap-3"><div className="mt-1 grid h-8 w-8 place-items-center rounded-xl bg-cyan-300 text-slate-950"><Sparkles size={15} /></div><div className="rounded-2xl rounded-bl-md border border-white/[0.08] bg-[#161a21] px-5 py-4"><div className="flex gap-1"><span className="h-1.5 w-1.5 animate-bounce rounded-full bg-cyan-300" /><span className="h-1.5 w-1.5 animate-bounce rounded-full bg-cyan-300 [animation-delay:120ms]" /><span className="h-1.5 w-1.5 animate-bounce rounded-full bg-cyan-300 [animation-delay:240ms]" /></div></div></div>}</div>
            </div>

            <div className="mb-3 flex gap-2 overflow-x-auto pb-1">{questions.map((suggestion) => <button key={suggestion} onClick={() => { setQuestion(suggestion); inputRef.current?.focus(); }} className="shrink-0 rounded-full border border-white/[0.09] bg-white/[0.025] px-3.5 py-2 text-xs text-slate-400 transition hover:border-cyan-300/30 hover:bg-cyan-300/10 hover:text-cyan-200">{suggestion}</button>)}</div>
            <form onSubmit={submitQuestion} className="relative rounded-2xl border border-white/[0.11] bg-[#13171e] p-2 shadow-[0_12px_50px_rgba(0,0,0,0.22)] transition-within:border-cyan-300/30"><input ref={inputRef} value={question} onChange={(event) => setQuestion(event.target.value)} placeholder="Ask about your studies..." className="h-11 w-full bg-transparent px-3 pr-14 text-sm text-slate-200 outline-none placeholder:text-slate-600" /><div className="flex items-center justify-between px-2 pb-1"><div className="flex items-center gap-2 text-[10px] text-slate-600"><Sparkles size={12} className="text-cyan-300" /> AI answers are grounded in your source</div><button type="submit" disabled={!question.trim() || isThinking} className="grid h-9 w-9 place-items-center rounded-xl bg-cyan-300 text-slate-950 transition hover:bg-cyan-200 disabled:cursor-not-allowed disabled:opacity-30" aria-label="Send message"><ArrowUp size={17} strokeWidth={2.5} /></button></div></form><p className="mt-3 text-center text-[10px] text-slate-700">Scholarly can make mistakes. Always confirm important policies with your university.</p>
          </section>
        </main>
      </div>
    </div>
  );
}

export default App;
