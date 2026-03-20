import { useState, useEffect, useRef } from 'react'

// ─── Types ───────────────────────────────────────────────────────────────────

interface LLMConfig {
  id: number
  name: string
  provider: string
  api_key: string
  base_url: string
  model_name: string
  report_language: string
  created_at: string
}

interface ReportItem {
  id: number
  is_job?: boolean
  job_id?: number
  report_id?: number
  topic: string
  version: string
  file_path: string
  format_style: string
  created_at: string
  status?: string
  error?: string
  llm_model?: string
  llm_base_url?: string
}

interface ReportDetail extends ReportItem {
  markdown: string
  round1: Record<string, unknown>
  round2: Record<string, unknown>
  round3: Record<string, unknown>
}

interface ChatMessage {
  id: number
  report_id: number
  role: 'user' | 'assistant' | 'system'
  content: string
  created_at: string
}

type Status = 'idle' | 'loading' | 'done' | 'error'
type TabKey = 'report' | 'round1' | 'round2' | 'round3'

const TABS: { key: TabKey; label: string }[] = [
  { key: 'report', label: '📄 研究报告' },
  { key: 'round1', label: '🔍 规划阶段' },
  { key: 'round2', label: '🔬 综合阶段' },
  { key: 'round3', label: '✅ 评审阶段' },
]

// ─── Markdown renderer ───────────────────────────────────────────────────────

type MarkdownTone = 'default' | 'inverse' | 'muted'

function renderInline(text: string, tone: MarkdownTone = 'default'): React.ReactNode[] {
  const inlineCodeClass = tone === 'inverse'
    ? 'bg-white/15 text-current px-1 py-0.5 rounded text-xs font-mono'
    : tone === 'muted'
      ? 'bg-white/80 text-current px-1 py-0.5 rounded text-xs font-mono'
      : 'bg-[#F1F1EF] text-current px-1 py-0.5 rounded text-xs font-mono'

  const parts = text.split(/(\*\*[^*]+\*\*|`[^`]+`)/g)
  return parts.map((part, i) => {
    if (part.startsWith('**') && part.endsWith('**'))
      return <strong key={i} className="font-semibold text-current">{part.slice(2, -2)}</strong>
    if (part.startsWith('`') && part.endsWith('`'))
      return <code key={i} className={inlineCodeClass}>{part.slice(1, -1)}</code>
    return <span key={i}>{part}</span>
  })
}

function MarkdownRenderer({ content, tone = 'default' }: { content: string, tone?: MarkdownTone }) {
  const lines = content.split('\n')
  const elements: React.ReactNode[] = []
  const codeBlockClass = tone === 'inverse'
    ? 'bg-black/20 rounded-md p-4 my-3 overflow-x-auto text-xs font-mono border border-white/10 text-current'
    : tone === 'muted'
      ? 'bg-white/80 rounded-md p-4 my-3 overflow-x-auto text-xs font-mono border border-[#E9E9E7] text-current'
      : 'bg-[#F7F6F3] rounded-md p-4 my-3 overflow-x-auto text-xs font-mono border border-[#E9E9E7] text-current'
  const blockquoteClass = tone === 'inverse'
    ? 'border-l-4 border-white/20 pl-3 my-2 text-sm text-current italic'
    : tone === 'muted'
      ? 'border-l-4 border-[#D9D8D4] pl-3 my-2 text-sm text-current italic'
      : 'border-l-4 border-[#E9E9E7] pl-3 my-2 text-sm text-current italic'
  const hrClass = tone === 'inverse' ? 'my-4 border-white/20' : 'my-4 border-[#E9E9E7]'
  const wrapperClass = tone === 'inverse' ? 'space-y-0.5 text-white' : tone === 'muted' ? 'space-y-0.5 text-[#9B9A97]' : 'space-y-0.5 text-[#37352F]'
  let i = 0
  while (i < lines.length) {
    const line = lines[i]
    if (line.startsWith('```')) {
      const codeLines: string[] = []
      i++
      while (i < lines.length && !lines[i].startsWith('```')) { codeLines.push(lines[i]); i++ }
      elements.push(
        <pre key={`code-${i}`} className={codeBlockClass}>
          {codeLines.join('\n')}
        </pre>
      )
    } else if (line.startsWith('# ')) {
      elements.push(<h1 key={i} className="text-2xl font-bold text-current mt-6 mb-2">{renderInline(line.slice(2), tone)}</h1>)
    } else if (line.startsWith('## ')) {
      elements.push(<h2 key={i} className="text-xl font-semibold text-current mt-5 mb-1.5">{renderInline(line.slice(3), tone)}</h2>)
    } else if (line.startsWith('### ')) {
      elements.push(<h3 key={i} className="text-base font-semibold text-current mt-4 mb-1">{renderInline(line.slice(4), tone)}</h3>)
    } else if (line.startsWith('- ') || line.startsWith('* ')) {
      elements.push(<li key={i} className="ml-5 text-sm text-current leading-relaxed list-disc">{renderInline(line.slice(2), tone)}</li>)
    } else if (/^\d+\.\s/.test(line)) {
      elements.push(<li key={i} className="ml-5 text-sm text-current leading-relaxed list-decimal">{renderInline(line.replace(/^\d+\.\s/, ''), tone)}</li>)
    } else if (line.startsWith('> ')) {
      elements.push(<blockquote key={i} className={blockquoteClass}>{renderInline(line.slice(2), tone)}</blockquote>)
    } else if (line.trim() === '---' || line.trim() === '***') {
      elements.push(<hr key={i} className={hrClass} />)
    } else if (line.trim() !== '') {
      elements.push(<p key={i} className="text-sm text-current leading-relaxed my-1">{renderInline(line, tone)}</p>)
    } else {
      elements.push(<div key={i} className="h-2" />)
    }
    i++
  }
  return <div className={wrapperClass}>{elements}</div>
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

function formatDate(iso: string) {
  return new Date(iso).toLocaleDateString('zh-CN', {
    month: 'short', day: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

// ─── App ─────────────────────────────────────────────────────────────────────

export default function App() {
  const [topic, setTopic] = useState('')
  const [format, setFormat] = useState('IEEE')
  const [reportLanguage, setReportLanguage] = useState('zh-CN')
  const [status, setStatus] = useState<Status>('idle')
  const [error, setError] = useState('')

  const [llmConfigs, setLlmConfigs] = useState<LLMConfig[]>([])
  const [selectedConfigId, setSelectedConfigId] = useState<number | null>(null)
  const [showConfigModal, setShowConfigModal] = useState(false)
  const [newConfigName, setNewConfigName] = useState('')
  const [newConfigProvider, setNewConfigProvider] = useState('OpenAI')
  const [newConfigApiKey, setNewConfigApiKey] = useState('')
  const [newConfigBaseUrl, setNewConfigBaseUrl] = useState('')
  const [newConfigModel, setNewConfigModel] = useState('')
  const [newConfigReportLanguage, setNewConfigReportLanguage] = useState('zh-CN')
  const [editingConfigId, setEditingConfigId] = useState<number | null>(null)

  const [history, setHistory] = useState<ReportItem[]>([])
  const [activeReport, setActiveReport] = useState<ReportDetail | null>(null)
  const [activeHistoryItem, setActiveHistoryItem] = useState<ReportItem | null>(null)
  const [activeTab, setActiveTab] = useState<TabKey>('report')
  const [generateLogs, setGenerateLogs] = useState<{ status: string, message: string, at: string }[]>([])
  const [activeJobId, setActiveJobId] = useState<number | null>(null)

  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [chatInput, setChatInput] = useState('')
  const [chatLoading, setChatLoading] = useState(false)
  const [regenLoading, setRegenLoading] = useState(false)
  const [lastInstruction, setLastInstruction] = useState('')

  const chatEndRef = useRef<HTMLDivElement>(null)

  const abortControllerRef = useRef<AbortController | null>(null)

  useEffect(() => { fetchHistory(); fetchLlmConfigs() }, [])
  useEffect(() => { chatEndRef.current?.scrollIntoView({ behavior: 'smooth' }) }, [messages])

  async function fetchHistory() {
    try {
      const res = await fetch('/api/history')
      if (res.ok) setHistory(await res.json())
    } catch { /* ignore */ }
  }

  // 当选中某个正在生成的 job 时，启动流
  useEffect(() => {
    if (activeJobId !== null) {
      startJobStream(activeJobId)
    }
    return () => {
      if (abortControllerRef.current) {
        abortControllerRef.current.abort()
        abortControllerRef.current = null
      }
    }
  }, [activeJobId])

  async function startJobStream(jobId: number) {
    if (abortControllerRef.current) {
      abortControllerRef.current.abort()
    }
    const controller = new AbortController()
    abortControllerRef.current = controller

    setGenerateLogs([{ status: 'init', message: '正在同步后台任务进度...', at: new Date().toISOString() }])
    setStatus('loading')
    setError('')

    try {
      const res = await fetch(`/api/jobs/${jobId}/stream`, { signal: controller.signal })
      if (!res.ok) throw new Error('流读取失败')
      const reader = res.body?.getReader()
      if (!reader) throw new Error('流读取失败')

      const decoder = new TextDecoder()
      let buffer = ''
      while (true) {
        const { done, value } = await reader.read()
        if (done) break
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split('\n\n')
        buffer = lines.pop() ?? ''

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            const dataStr = line.slice(6)
            try {
              const parsed = JSON.parse(dataStr)
              if (parsed.status === 'error') {
                throw new Error(parsed.message)
              } else if (parsed.status === 'done') {
                const reportId = parsed.report_id
                fetchHistory()
                setActiveJobId(null)
                if (reportId) {
                  await fetchReportData(reportId)
                  setStatus('done')
                }
              } else {
                setGenerateLogs(prev => [...prev, { status: parsed.status, message: parsed.message, at: new Date().toISOString() }])
              }
            } catch (e: any) {
              if (e.message && e.message !== 'Unexpected end of JSON input') {
                throw e
              }
            }
          }
        }
      }
    } catch (e: any) {
      if (e.name !== 'AbortError') {
        setError(e.message)
        setStatus('error')
        setActiveJobId(null)
      }
    }
  }

  async function fetchReportData(id: number) {
    const [rRes, mRes] = await Promise.all([
      fetch(`/api/report/${id}`),
      fetch(`/api/report/${id}/messages`),
    ])
    if (rRes.ok) {
      setActiveReport(await rRes.json())
      setActiveTab('report')
    }
    if (mRes.ok) setMessages(await mRes.json())
    else setMessages([])
  }

  async function openHistoryItem(item: ReportItem) {
    setActiveHistoryItem(item)
    if (item.is_job && item.status !== 'done') {
      if (item.status === 'error') {
        setStatus('error')
        setError(item.error || '任务执行失败')
        setActiveReport(null)
      } else {
        setActiveReport(null)
        setMessages([])
        setActiveJobId(item.job_id!)
      }
    } else {
      setActiveJobId(null)
      setStatus('done')
      const targetId = item.is_job ? item.report_id : item.id
      if (targetId) {
        await fetchReportData(targetId)
      }
    }
  }

  async function handleRetryJob() {
    if (!activeHistoryItem || !activeHistoryItem.job_id) return
    const jobId = activeHistoryItem.job_id

    setStatus('loading')
    setError('')
    setActiveReport(null)
    setMessages([])
    setGenerateLogs([{ status: 'init', message: '正在请求重试后台任务...', at: new Date().toISOString() }])

    try {
      const res = await fetch(`/api/jobs/${jobId}/retry`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({}),
      })
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail ?? '重试失败')
      }

      fetchHistory()
      setActiveJobId(jobId)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
      setStatus('error')
    }
  }
  async function fetchLlmConfigs() {
    try {
      const res = await fetch('/api/llm-configs')
      if (res.ok) {
        const configs: LLMConfig[] = await res.json()
        setLlmConfigs(configs)
        if (configs.length > 0) setSelectedConfigId(prev => prev ?? configs[0].id)
      }
    } catch { /* ignore */ }
  }

  async function handleSaveConfig() {
    if (!newConfigName.trim() || !newConfigApiKey.trim() || !newConfigModel.trim()) return

    const body = {
      name: newConfigName.trim(),
      provider: newConfigProvider,
      api_key: newConfigApiKey.trim(),
      base_url: newConfigBaseUrl.trim(),
      model_name: newConfigModel.trim(),
      report_language: newConfigReportLanguage,
    }

    if (editingConfigId !== null) {
      const res = await fetch(`/api/llm-configs/${editingConfigId}`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (res.ok) {
        const updated: LLMConfig = await res.json()
        setLlmConfigs(prev => prev.map(c => c.id === updated.id ? updated : c))
        if (selectedConfigId === updated.id) setSelectedConfigId(updated.id)

        setNewConfigName('')
        setNewConfigApiKey('')
        setNewConfigBaseUrl('')
        setNewConfigModel('')
        setNewConfigProvider('OpenAI')
        setNewConfigReportLanguage('zh-CN')
        setEditingConfigId(null)
      }
    } else {
      const res = await fetch('/api/llm-configs', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      })
      if (res.ok) {
        const saved: LLMConfig = await res.json()
        setLlmConfigs(prev => [...prev, saved])
        setSelectedConfigId(saved.id)
        setNewConfigName('')
        setNewConfigApiKey('')
        setNewConfigBaseUrl('')
        setNewConfigModel('')
        setNewConfigProvider('OpenAI')
        setNewConfigReportLanguage('zh-CN')
      }
    }
  }

  function handleEditConfig(config: LLMConfig) {
    setNewConfigName(config.name)
    setNewConfigProvider(config.provider)
    setNewConfigApiKey(config.api_key)
    setNewConfigBaseUrl(config.base_url || '')
    setNewConfigModel(config.model_name)
    setNewConfigReportLanguage(config.report_language || 'zh-CN')
    setEditingConfigId(config.id)
  }

  function handleCancelEdit() {
    setNewConfigName('')
    setNewConfigApiKey('')
    setNewConfigBaseUrl('')
    setNewConfigModel('')
    setNewConfigProvider('OpenAI')
    setNewConfigReportLanguage('zh-CN')
    setEditingConfigId(null)
  }

  function handleCloseConfigModal() {
    handleCancelEdit()
    setShowConfigModal(false)
  }

  async function handleDeleteConfig(id: number) {
    await fetch(`/api/llm-configs/${id}`, { method: 'DELETE' })
    setLlmConfigs(prev => prev.filter(c => c.id !== id))
    if (selectedConfigId === id) setSelectedConfigId(llmConfigs.find(c => c.id !== id)?.id ?? null)
  }

  function getSelectedConfig(): LLMConfig | undefined {
    return llmConfigs.find(c => c.id === selectedConfigId)
  }

  async function handleGenerate(retryTopic?: string) {
    const finalTopic = retryTopic ?? topic
    if (!finalTopic.trim()) return
    setTopic(finalTopic)
    const cfg = getSelectedConfig()
    setStatus('loading')
    setError('')
    setActiveReport(null)
    setMessages([])
    setActiveHistoryItem(null)
    setActiveJobId(null)
    setGenerateLogs([{ status: 'init', message: '请求已发出，正在与后端建立连接并唤起大模型...', at: new Date().toISOString() }])
    try {
      const res = await fetch('/api/report', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          topic: finalTopic,
          format_style: format,
          api_key: cfg?.api_key || undefined,
          model: cfg?.model_name || undefined,
          base_url: cfg?.base_url || undefined,
          report_language: reportLanguage,
        }),
      })
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail ?? '生成失败')
      }

      const data = await res.json()
      if (data.job_id) {
        fetchHistory()
        setActiveJobId(data.job_id)
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
      setStatus('error')
    }
  }

  async function handleChat() {
    if (!chatInput.trim() || !activeReport) return
    const cfg = getSelectedConfig()
    const userMsg = chatInput.trim()
    setChatInput('')
    setChatLoading(true)
    const temp: ChatMessage = { id: Date.now(), report_id: activeReport.id, role: 'user', content: userMsg, created_at: new Date().toISOString() }
    setMessages(prev => [...prev, temp])
    try {
      const res = await fetch(`/api/report/${activeReport.id}/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg, api_key: cfg?.api_key || undefined, model: cfg?.model_name || undefined, base_url: cfg?.base_url || undefined }),
      })
      if (res.ok) {
        const reply = await res.json()
        setMessages(prev => [...prev, reply])
      }
    } finally {
      setChatLoading(false)
    }
  }

  async function handleRegenerate(retryInstruction?: string) {
    const instruction = retryInstruction || chatInput.trim()
    if (!activeReport || !instruction) return
    const cfg = getSelectedConfig()
    setChatInput('')
    setLastInstruction(instruction)
    setRegenLoading(true)
    setGenerateLogs([{ status: 'init', message: '请求已发出，正在与后端建立连接并唤起大模型...', at: new Date().toISOString() }])
    try {
      const res = await fetch(`/api/report/${activeReport.id}/regenerate`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ instruction, api_key: cfg?.api_key || undefined, model: cfg?.model_name || undefined, base_url: cfg?.base_url || undefined }),
      })
      if (!res.ok) {
        const data = await res.json()
        throw new Error(data.detail ?? '重新生成失败')
      }

      const data = await res.json()
      if (data.job_id) {
        fetchHistory()
        setActiveJobId(data.job_id)
      }
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : String(e))
    } finally {
      setRegenLoading(false)
    }
  }

  function handleNewReport() {
    setActiveHistoryItem(null)
    setActiveReport(null)
    setActiveJobId(null)
    setStatus('idle')
    setError('')
    setTopic('')
    setMessages([])
  }

  const isGenerating = status === 'loading'

  return (
    <div className="flex h-screen bg-white font-sans overflow-hidden">

      {/* ── Sidebar: History ────────────────────────────────────────────────── */}
      <aside className="w-56 shrink-0 border-r border-[#E9E9E7] flex flex-col bg-[#FBFBFA]">
        <div className="px-4 py-3 border-b border-[#E9E9E7]">
          <div className="flex items-center gap-2">
            <span className="text-sm">📄</span>
            <span className="text-sm font-semibold text-[#37352F]">研究前置调研</span>
          </div>
        </div>
        <div className="px-3 pt-3">
          <button
            onClick={handleNewReport}
            className="w-full flex items-center justify-center gap-2 py-1.5 rounded-md border border-[#E9E9E7] bg-white text-xs font-medium text-[#37352F] hover:bg-[#F1F1EF] transition-colors"
          >
            + 新建报告
          </button>
        </div>
        <div className="px-3 pt-4 pb-1">
          <p className="text-[10px] font-semibold text-[#9B9A97] uppercase tracking-wider px-1">历史报告</p>
        </div>
        <div className="flex-1 overflow-y-auto">
          {history.length === 0 && (
            <p className="text-xs text-[#C7C6C4] px-4 py-3">暂无历史记录</p>
          )}
          {history.map(h => (
            <button
              key={h.id}
              onClick={() => openHistoryItem(h)}
              className={`w-full text-left px-4 py-2.5 hover:bg-[#F1F1EF] transition-colors border-b border-[#F1F1EF] ${activeHistoryItem?.id === h.id ? 'bg-[#F1F1EF]' : ''
                }`}
            >
              <p className="text-xs font-medium text-[#37352F] truncate">
                {h.is_job && h.status === 'running' ? '⏳ ' : ''}
                {h.is_job && h.status === 'error' ? '❌ ' : ''}
                {h.topic}
              </p>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className="text-[10px] font-mono text-[#9B9A97]">{h.version}</span>
                <span className="text-[10px] text-[#C7C6C4]">·</span>
                <span className="text-[10px] text-[#9B9A97]">{formatDate(h.created_at)}</span>
              </div>
            </button>
          ))}
        </div>
      </aside>

      {/* ── Center: report + form ───────────────────────────────────────────── */}
      <div className="flex-1 flex flex-col overflow-hidden">

        {/* Header + form */}
        <div className="border-b border-[#E9E9E7] py-5 space-y-3 shrink-0">
          <div className="max-w-xl mx-auto px-8">
            <div>
              <h1 className="text-2xl font-bold text-[#37352F] tracking-tight">研究前置调研报告</h1>
              <p className="text-xs text-[#9B9A97] mt-0.5">输入研究方向，多轮 LLM 讨论 + 网络搜索，自动生成前置调研报告</p>
            </div>
            <div className="flex items-start gap-3 mt-3">
              <span className="text-xs text-[#9B9A97] w-16 shrink-0 pt-2">研究主题</span>
              <textarea
                value={topic}
                onChange={e => setTopic(e.target.value)}
                onKeyDown={e => { if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleGenerate() }}
                rows={4}
                placeholder="例如：基于多模态数据的抑郁症风险预测"
                className="flex-1 text-sm px-3 py-2 rounded-md border border-[#E9E9E7] bg-white focus:outline-none focus:ring-1 focus:ring-[#37352F] text-[#37352F] placeholder:text-[#C7C6C4] resize-none"
              />
            </div>
            {/* LLM config selector */}
            <div className="flex items-center gap-3 mt-3">
              <span className="text-xs text-[#9B9A97] w-16 shrink-0">LLM 配置</span>
              <select
                value={selectedConfigId ?? ''}
                onChange={e => setSelectedConfigId(Number(e.target.value))}
                className="flex-1 text-sm px-3 py-1.5 rounded-md border border-[#E9E9E7] bg-white focus:outline-none focus:ring-1 focus:ring-[#37352F] text-[#37352F]"
              >
                {llmConfigs.length === 0 && <option value="">— 尚未配置，请点击管理 —</option>}
                {llmConfigs.map(c => (
                  <option key={c.id} value={c.id}>
                    [{c.provider}] {c.name}  ·  {c.model_name}
                  </option>
                ))}
              </select>
              <button
                onClick={() => setShowConfigModal(true)}
                className="px-3 py-1.5 rounded-md border border-[#E9E9E7] text-xs text-[#37352F] hover:bg-[#F1F1EF] transition-colors shrink-0"
              >
                管理
              </button>
            </div>
            {/* Report language selector */}
            <div className="flex items-center gap-3 mt-3">
              <span className="text-xs text-[#9B9A97] w-16 shrink-0">报告语言</span>
              <select
                value={reportLanguage}
                onChange={e => setReportLanguage(e.target.value)}
                className="text-sm px-3 py-1.5 rounded-md border border-[#E9E9E7] bg-white focus:outline-none focus:ring-1 focus:ring-[#37352F] text-[#37352F]"
              >
                <option value="zh-CN">中文简体</option>
                <option value="en">English</option>
              </select>
            </div>
            <div className="flex items-center gap-3 mt-3">
              <span className="text-xs text-[#9B9A97] w-16 shrink-0">输出格式</span>
              <select
                value={format}
                onChange={e => setFormat(e.target.value)}
                className="text-sm px-3 py-1.5 rounded-md border border-[#E9E9E7] bg-white focus:outline-none focus:ring-1 focus:ring-[#37352F] text-[#37352F]"
              >
                <option>IEEE</option>
                <option>APA</option>
                <option>Chicago</option>
              </select>
              <button
                onClick={() => handleGenerate()}
                disabled={isGenerating || !topic.trim()}
                className="ml-auto px-4 py-1.5 rounded-md bg-[#37352F] text-white text-sm font-medium hover:bg-[#1a1917] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                {isGenerating ? '生成中…' : '生成报告  ⌘↵'}
              </button>
            </div>
          </div>
        </div>

        {/* Report content */}
        <div className="flex-1 overflow-y-auto">
          {!activeReport && !isGenerating && status !== 'error' && (
            <div className="flex flex-col items-center justify-center h-full gap-2 text-center">
              <span className="text-4xl">🔬</span>
              <p className="text-sm text-[#9B9A97]">输入研究主题并点击「生成报告」</p>
              <p className="text-xs text-[#C7C6C4]">或从左侧历史记录中选择已有报告</p>
            </div>
          )}

          {isGenerating && (
            <div className="flex flex-col items-center justify-center h-full gap-4 max-w-3xl mx-auto w-full px-8">
              <div className="flex items-center gap-3">
                <div className="w-6 h-6 border-2 border-[#37352F] border-t-transparent rounded-full animate-spin" />
                <div>
                  <p className="text-sm font-semibold text-[#37352F]">正在生成报告，通常需要 1–3 分钟…</p>
                  <p className="text-xs text-[#9B9A97] mt-0.5">下方显示从查询、网页抓取到多轮生成的完整链路进度</p>
                  {(() => {
                    const cfg = getSelectedConfig()
                    const modelName = activeHistoryItem?.llm_model || cfg?.model_name
                    const baseUrl = activeHistoryItem?.llm_base_url || cfg?.base_url
                    const provider = activeHistoryItem?.llm_model ? 'Job Config' : (cfg?.provider || 'Unknown')

                    if (modelName) {
                      return (
                        <p className="text-xs text-[#9B9A97] mt-1 font-mono">
                          [{provider}] {modelName}{baseUrl ? ' · ' + baseUrl : ''}
                        </p>
                      )
                    }
                    return null
                  })()}
                </div>
              </div>

              <div className="w-full bg-[#FBFBFA] border border-[#E9E9E7] rounded-md p-4 space-y-2 h-96 overflow-y-auto">
                {generateLogs.length === 0 && (
                  <p className="text-xs text-[#9B9A97]">等待服务器响应...</p>
                )}
                {generateLogs.map((log, i) => (
                  <div key={i} className="flex gap-2 text-xs">
                    <span className="text-[#9B9A97] shrink-0 font-mono">[{new Date(log.at).toLocaleTimeString('zh-CN', { hour12: false })}]</span>
                    <span className="text-[#37352F]">{log.message}</span>
                  </div>
                ))}
                <div ref={el => el?.scrollIntoView()} />
              </div>
            </div>
          )}

          {status === 'error' && (
            <div className="max-w-3xl mx-auto px-8 mt-6 rounded-md bg-red-50 border border-red-200 px-4 py-3 text-sm text-red-600">
              <div className="flex items-center justify-between gap-3">
                <div>
                  <strong>生成失败：</strong>{error}
                </div>
                <div className="shrink-0">
                  {activeHistoryItem?.job_id ? (
                    <button
                      onClick={() => handleRetryJob()}
                      disabled={isGenerating}
                      className="px-3 py-1.5 rounded-md border border-red-300 bg-white text-xs text-red-600 hover:bg-red-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    >
                      重试当前任务
                    </button>
                  ) : activeReport && lastInstruction ? (
                    <button
                      onClick={() => handleRegenerate(lastInstruction)}
                      disabled={regenLoading}
                      className="px-3 py-1.5 rounded-md border border-red-300 bg-white text-xs text-red-600 hover:bg-red-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    >
                      {regenLoading ? '重试中…' : '重试'}
                    </button>
                  ) : (
                    <button
                      onClick={() => handleGenerate(activeHistoryItem?.topic || topic)}
                      disabled={isGenerating || !(activeHistoryItem?.topic || topic).trim()}
                      className="px-3 py-1.5 rounded-md border border-red-300 bg-white text-xs text-red-600 hover:bg-red-50 disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                    >
                      重试
                    </button>
                  )}
                </div>
              </div>
            </div>
          )}

          {activeReport && !isGenerating && (
            <div className="flex flex-col h-full">
              {/* Tabs */}
              <div className="flex items-center gap-1 w-full px-8 pt-4 border-b border-[#E9E9E7] shrink-0">
                {TABS.map(t => (
                  <button
                    key={t.key}
                    onClick={() => setActiveTab(t.key)}
                    className={`px-3 py-1.5 text-xs rounded-t-md border-b-2 transition-colors ${activeTab === t.key
                      ? 'border-[#37352F] text-[#37352F] font-semibold'
                      : 'border-transparent text-[#9B9A97] hover:text-[#37352F]'
                      }`}
                  >
                    {t.label}
                  </button>
                ))}
                <button
                  onClick={() => navigator.clipboard.writeText(activeReport.markdown)}
                  className="ml-auto text-xs text-[#9B9A97] hover:text-[#37352F] px-2 py-1"
                  title="复制 Markdown 到剪贴板"
                >
                  复制 MD
                </button>
                <span className="text-[10px] font-mono text-[#C7C6C4] ml-1">{activeReport.version}</span>
              </div>
              {/* Tab content */}
              <div className="flex-1 overflow-y-auto">
                <div className="px-8 py-5">
                  {activeTab === 'report' ? (
                    <MarkdownRenderer content={activeReport.markdown} />
                  ) : (
                    <pre className="text-xs font-mono text-[#9B9A97] whitespace-pre-wrap break-words leading-relaxed">
                      {JSON.stringify(activeReport[activeTab], null, 2)}
                    </pre>
                  )}
                </div>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* ── Right panel: Chat / Regenerate ──────────────────────────────────── */}
      {activeReport && (
        <aside className="w-[400px] shrink-0 border-l border-[#E9E9E7] flex flex-col bg-[#FBFBFA]">
          <div className="px-4 py-3 border-b border-[#E9E9E7]">
            <p className="text-sm font-semibold text-[#37352F]">💬 对话</p>
            <p className="text-[10px] text-[#9B9A97] mt-0.5">输入指令后点击「发送」微调，或点击「重新生成」生成新版本</p>
          </div>

          {/* Chat messages */}
          <div className="flex-1 overflow-y-auto px-4 py-3 space-y-3">
            {messages.length === 0 && (
              <p className="text-xs text-[#C7C6C4] text-center mt-4">暂无对话记录</p>
            )}
            {messages.map(m => (
              <div key={m.id} className={`flex flex-col gap-0.5 ${m.role === 'user' ? 'items-end' : 'items-start'
                }`}>
                <span className="text-[10px] text-[#C7C6C4]">
                  {m.role === 'user' ? '你' : m.role === 'system' ? '系统' : 'AI'}
                </span>
                <div className={`max-w-[90%] rounded-lg px-3 py-2 text-xs leading-relaxed ${m.role === 'user'
                  ? 'bg-[#37352F] text-white'
                  : m.role === 'system'
                    ? 'bg-[#F1F1EF] text-[#9B9A97] italic'
                    : 'bg-white border border-[#E9E9E7] text-[#37352F]'
                  }`}>
                  <MarkdownRenderer
                    content={m.content}
                    tone={m.role === 'user' ? 'inverse' : m.role === 'system' ? 'muted' : 'default'}
                  />
                  {m.role === 'assistant' && (
                    <button
                      onClick={() => navigator.clipboard.writeText(m.content)}
                      className="mt-1.5 text-[10px] text-[#9B9A97] hover:text-[#37352F] transition-colors"
                      title="复制 Markdown"
                    >
                      复制 MD
                    </button>
                  )}
                </div>
              </div>
            ))}
            {(chatLoading || regenLoading) && (
              <div className="flex flex-col gap-2">
                <div className="flex items-center gap-2 text-xs text-[#9B9A97]">
                  <div className="w-3 h-3 border border-[#9B9A97] border-t-transparent rounded-full animate-spin" />
                  {regenLoading ? '重新生成中…' : '思考中…'}
                </div>
                {regenLoading && generateLogs.length > 0 && (
                  <div className="text-[10px] text-[#AEADA9] italic pl-5 border-l-2 border-[#E9E9E7] ml-1.5 py-1">
                    {generateLogs[generateLogs.length - 1].message}
                  </div>
                )}
              </div>
            )}
            <div ref={chatEndRef} />
          </div>

          {/* Input area */}
          <div className="border-t border-[#E9E9E7] p-3 space-y-2">
            <textarea
              value={chatInput}
              onChange={e => setChatInput(e.target.value)}
              onKeyDown={e => { if (e.key === 'Enter' && !e.shiftKey && !e.metaKey) { e.preventDefault(); handleChat() } }}
              rows={3}
              placeholder="输入微调指令…（Enter 发送，Shift+Enter 换行）"
              className="w-full text-xs px-3 py-2 rounded-md border border-[#E9E9E7] bg-white focus:outline-none focus:ring-1 focus:ring-[#37352F] text-[#37352F] placeholder:text-[#C7C6C4] resize-none"
            />
            <div className="flex gap-2">
              <button
                onClick={handleChat}
                disabled={chatLoading || regenLoading || !chatInput.trim()}
                className="flex-1 py-1.5 rounded-md border border-[#E9E9E7] text-xs text-[#37352F] hover:bg-[#F1F1EF] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                发送
              </button>
              <button
                onClick={() => handleRegenerate()}
                disabled={chatLoading || regenLoading || !chatInput.trim()}
                className="flex-1 py-1.5 rounded-md bg-[#37352F] text-white text-xs font-medium hover:bg-[#1a1917] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
              >
                🔄 重新生成
              </button>
            </div>
            <p className="text-[10px] text-[#C7C6C4]">
              重新生成将保存为新版本，旧版本 .md 文件保留
            </p>
          </div>
        </aside>
      )}

      {/* ── LLM Config Modal ─────────────────────────────────────────────────── */}
      {showConfigModal && (
        <div className="fixed inset-0 bg-black/40 z-50 flex items-center justify-center">
          <div className="bg-white rounded-xl shadow-2xl w-[480px] max-h-[80vh] overflow-y-auto p-6 space-y-5" onClick={e => e.stopPropagation()}>
            <h2 className="text-base font-semibold text-[#37352F]">管理 LLM 配置</h2>

            {/* Existing configs */}
            {llmConfigs.length > 0 && (
              <div className="space-y-2">
                {llmConfigs.map(c => (
                  <div key={c.id} className="flex items-center gap-2 px-3 py-2 rounded-md border border-[#E9E9E7] bg-[#FBFBFA]">
                    <div className="flex-1 min-w-0">
                      <p className="text-sm font-medium text-[#37352F] truncate">{c.name}</p>
                      <p className="text-[10px] text-[#9B9A97] truncate">[{c.provider}] {c.model_name}{c.base_url ? ' · ' + c.base_url : ''}</p>
                    </div>
                    <button
                      onClick={() => handleEditConfig(c)}
                      className="text-[#9B9A97] hover:text-[#37352F] text-xs px-2 py-1 rounded hover:bg-[#F1F1EF] transition-colors shrink-0"
                    >
                      编辑
                    </button>
                    <button
                      onClick={() => handleDeleteConfig(c.id)}
                      className="text-[#9B9A97] hover:text-red-500 text-xs px-2 py-1 rounded hover:bg-red-50 transition-colors shrink-0"
                    >
                      删除
                    </button>
                  </div>
                ))}
              </div>
            )}

            {/* Add new config form */}
            <div className="border-t border-[#E9E9E7] pt-4 space-y-3">
              <div className="flex items-center justify-between gap-3">
                <p className="text-xs font-semibold text-[#9B9A97] uppercase tracking-wider">
                  {editingConfigId !== null ? '编辑配置' : '添加新配置'}
                </p>
                {editingConfigId !== null && (
                  <button
                    onClick={handleCancelEdit}
                    className="text-xs text-[#9B9A97] hover:text-[#37352F]"
                  >
                    取消编辑
                  </button>
                )}
              </div>
              <div className="space-y-2">
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[#9B9A97] w-24 shrink-0">配置名称</span>
                  <input
                    value={newConfigName}
                    onChange={e => setNewConfigName(e.target.value)}
                    placeholder="例如：Gemini Flash"
                    className="flex-1 text-sm px-2.5 py-1.5 rounded-md border border-[#E9E9E7] focus:outline-none focus:ring-1 focus:ring-[#37352F]"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[#9B9A97] w-24 shrink-0">Provider</span>
                  <select
                    value={newConfigProvider}
                    onChange={e => {
                      const p = e.target.value
                      setNewConfigProvider(p)
                      if (p === 'Ollama') {
                        if (!newConfigBaseUrl) setNewConfigBaseUrl('http://localhost:11434/v1')
                        if (!newConfigApiKey) setNewConfigApiKey('ollama')
                      }
                    }}
                    className="flex-1 text-sm px-2.5 py-1.5 rounded-md border border-[#E9E9E7] focus:outline-none focus:ring-1 focus:ring-[#37352F] bg-white"
                  >
                    <option>OpenAI</option>
                    <option>Gemini</option>
                    <option>Ollama</option>
                    <option>Other</option>
                  </select>
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[#9B9A97] w-24 shrink-0">API Key</span>
                  <input
                    type="password"
                    value={newConfigApiKey}
                    onChange={e => setNewConfigApiKey(e.target.value)}
                    placeholder={newConfigProvider === 'Ollama' ? '留空或填 ollama（本地无需 key）' : 'sk-… 或 AIza…'}
                    className="flex-1 text-sm px-2.5 py-1.5 rounded-md border border-[#E9E9E7] focus:outline-none focus:ring-1 focus:ring-[#37352F]"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[#9B9A97] w-24 shrink-0">Base URL</span>
                  <input
                    value={newConfigBaseUrl}
                    onChange={e => setNewConfigBaseUrl(e.target.value)}
                    placeholder={newConfigProvider === 'Ollama' ? 'http://localhost:11434/v1' : '留空则使用默认端点'}
                    className="flex-1 text-sm px-2.5 py-1.5 rounded-md border border-[#E9E9E7] focus:outline-none focus:ring-1 focus:ring-[#37352F]"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[#9B9A97] w-24 shrink-0">Model Name</span>
                  <input
                    value={newConfigModel}
                    onChange={e => setNewConfigModel(e.target.value)}
                    placeholder={newConfigProvider === 'Ollama' ? 'llama3 / qwen2.5 / mistral…' : 'gpt-4o / gemini-2.5-flash…'}
                    className="flex-1 text-sm px-2.5 py-1.5 rounded-md border border-[#E9E9E7] focus:outline-none focus:ring-1 focus:ring-[#37352F]"
                  />
                </div>
                <div className="flex items-center gap-2">
                  <span className="text-xs text-[#9B9A97] w-24 shrink-0">报告语言</span>
                  <select
                    value={newConfigReportLanguage}
                    onChange={e => setNewConfigReportLanguage(e.target.value)}
                    className="flex-1 text-sm px-2.5 py-1.5 rounded-md border border-[#E9E9E7] focus:outline-none focus:ring-1 focus:ring-[#37352F] bg-white"
                  >
                    <option value="zh-CN">中文简体</option>
                    <option value="en">英文</option>
                  </select>
                </div>
              </div>
              <div className="flex justify-end gap-2 pt-1">
                <button
                  onClick={handleCloseConfigModal}
                  className="px-4 py-1.5 rounded-md border border-[#E9E9E7] text-sm text-[#37352F] hover:bg-[#F1F1EF] transition-colors"
                >
                  关闭
                </button>
                <button
                  onClick={handleSaveConfig}
                  disabled={!newConfigName.trim() || (newConfigProvider !== 'Ollama' && !newConfigApiKey.trim()) || !newConfigModel.trim()}
                  className="px-4 py-1.5 rounded-md bg-[#37352F] text-white text-sm font-medium hover:bg-[#1a1917] disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
                >
                  {editingConfigId !== null ? '更新' : '保存'}
                </button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

