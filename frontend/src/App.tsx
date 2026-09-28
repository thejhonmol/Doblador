import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Upload, FileVideo, Loader2, CheckCircle2, Mic, TextSelect, FileAudio, Film, Smile, ChevronDown, ChevronUp, Sparkles, Users, BarChart3, Download } from 'lucide-react';
import './index.css';

const API_URL = 'http://localhost:3000/api';

type JobStatus = 'pending' | 'processing' | 'completed' | 'failed';

interface Stage {
  id: string;
  stage: string;
  status: string;
  started_at: string;
  finished_at: string;
  download_url?: string;
  error?: string;
}

interface Job {
  id: string;
  status: JobStatus;
  target_lang: string;
  voice_mode?: string;
}

interface Speaker {
  speaker_label: string;
  fish_reference_id: string;
  target_lang: string;
  voice_mode?: string;
}

interface SegmentItem {
  speaker_label: string;
  start_ms: number;
  end_ms: number;
  source_text: string;
  translated_text?: string;
  emotion?: string;
  emotion_confidence?: number;
}

interface SyncReport {
  score: number;
  grade: string;
  summary: {
    segments_total: number;
    max_start_drift_ms: number;
    regenerated_segments: number;
    compressed_segments: number;
    clipped_segments: number;
  };
}

const EMOTION_MAP: Record<string, { label: string; emoji: string; color: string; bg: string }> = {
  happy: { label: 'Happy', emoji: '😊', color: '#4ade80', bg: 'rgba(34, 197, 94, 0.15)' },
  sad: { label: 'Sad', emoji: '😢', color: '#60a5fa', bg: 'rgba(59, 130, 246, 0.15)' },
  angry: { label: 'Angry', emoji: '😠', color: '#f87171', bg: 'rgba(239, 68, 68, 0.15)' },
  surprised: { label: 'Surprised', emoji: '😲', color: '#fbbf24', bg: 'rgba(245, 158, 11, 0.15)' },
  fearful: { label: 'Fearful', emoji: '😨', color: '#c084fc', bg: 'rgba(168, 85, 247, 0.15)' },
  disgust: { label: 'Disgust', emoji: '🤢', color: '#2dd4bf', bg: 'rgba(20, 184, 166, 0.15)' },
  neutral: { label: 'Neutral', emoji: '😐', color: '#94a3b8', bg: 'rgba(100, 116, 139, 0.15)' },
};

function formatTimeMs(ms: number): string {
  const totalSec = Math.floor(ms / 1000);
  const mins = Math.floor(totalSec / 60);
  const secs = totalSec % 60;
  return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [lang, setLang] = useState('Spanish');
  const [voiceMode, setVoiceMode] = useState<'clone' | 'preset'>('clone');
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [stages, setStages] = useState<Stage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [segmentCount, setSegmentCount] = useState(0);
  const [emotionsSummary, setEmotionsSummary] = useState<Record<string, number>>({});
  const [segments, setSegments] = useState<SegmentItem[]>([]);
  const [syncReport, setSyncReport] = useState<SyncReport | null>(null);
  const [showTimeline, setShowTimeline] = useState(false);

  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleDrag = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setIsDragging(true);
    } else if (e.type === 'dragleave') {
      setIsDragging(false);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      setFile(e.dataTransfer.files[0]);
    }
  };

  const handleUpload = async () => {
    if (!file) return;
    setIsUploading(true);
    setError(null);
    setJobId(null);
    setJob(null);
    setStages([]);
    setEmotionsSummary({});
    setSegments([]);
    setSyncReport(null);
    setShowTimeline(false);

    const formData = new FormData();
    formData.append('video', file);
    formData.append('targetLang', lang);
    formData.append('voiceMode', voiceMode);

    try {
      const res = await axios.post(`${API_URL}/upload`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' }
      });
      const newJobId = res.data.jobId;
      setJobId(newJobId);

      // Immediate fetch so UI updates immediately
      try {
        const jobRes = await axios.get(`${API_URL}/jobs/${newJobId}`);
        setJob(jobRes.data.job);
        setStages(jobRes.data.stages || []);
        setSpeakers(jobRes.data.speakers || []);
        setSegmentCount(jobRes.data.segmentCount || 0);
        setEmotionsSummary(jobRes.data.emotionsSummary || {});
        setSegments(jobRes.data.segments || []);
        setSyncReport(jobRes.data.syncReport || null);
      } catch {}
    } catch (err: any) {
      setError(err.response?.data?.error || err.message || 'Upload failed');
    } finally {
      setIsUploading(false);
    }
  };

  useEffect(() => {
    if (!jobId) return;

    const interval = setInterval(async () => {
      try {
        const res = await axios.get(`${API_URL}/jobs/${jobId}`);
        setJob(res.data.job);
        setStages(res.data.stages || []);
        setSpeakers(res.data.speakers || []);
        setSegmentCount(res.data.segmentCount || 0);
        setEmotionsSummary(res.data.emotionsSummary || {});
        setSegments(res.data.segments || []);
        setSyncReport(res.data.syncReport || null);

        if (res.data.job.status === 'completed' || res.data.job.status === 'failed') {
          clearInterval(interval);
        }
      } catch (err) {
        console.error('Error fetching job', err);
      }
    }, 2000);

    return () => clearInterval(interval);
  }, [jobId]);

  const STAGE_CONFIG: Record<string, { label: string, icon: any }> = {
    'extract': { label: 'Audio Extraction', icon: Mic },
    'translate': { label: 'AI Translation (Gemini)', icon: TextSelect },
    'tts': { label: 'Speech Generation', icon: FileAudio },
    'assemble': { label: 'Final Muxing', icon: Film },
  };

  // Mock stages to display the full pipeline even if empty
  const displayStages = ['extract', 'translate', 'tts', 'assemble'].map(key => {
    const found = stages.find(s => s.stage === key);
    return {
      key,
      label: STAGE_CONFIG[key].label,
      Icon: STAGE_CONFIG[key].icon,
      status: found ? found.status : 'pending',
      error: found?.error,
      download_url: found?.download_url
    };
  });

  const assembleStage = stages.find(s => s.stage === 'assemble' && s.status === 'completed');
  const finalVideoUrl = assembleStage?.download_url ? `${API_URL.replace(/\/api$/, '')}${assembleStage.download_url}` : null;

  return (
    <div className="app-container">
      <div className="glass-panel">
        <div className="header">
          <h1>AI Dubbing Studio</h1>
          <p>Professional video dubbing: translated speech, re-voiced per speaker, timed to the original.</p>
        </div>

        {!jobId && (
          <>
            <div 
              className={`upload-zone ${isDragging ? 'drag-active' : ''}`}
              onDragEnter={handleDrag}
              onDragLeave={handleDrag}
              onDragOver={handleDrag}
              onDrop={handleDrop}
              onClick={() => fileInputRef.current?.click()}
            >
              <input 
                type="file" 
                ref={fileInputRef} 
                className="hidden" 
                style={{ display: 'none' }}
                accept="video/*"
                onChange={(e) => e.target.files && setFile(e.target.files[0])}
              />
              
              {file ? (
                <>
                  <FileVideo className="upload-icon" size={48} />
                  <div className="upload-text">{file.name}</div>
                  <div className="upload-hint">{(file.size / (1024 * 1024)).toFixed(2)} MB</div>
                </>
              ) : (
                <>
                  <Upload className="upload-icon" size={48} />
                  <div className="upload-text">Drag & Drop video here</div>
                  <div className="upload-hint">or click to browse</div>
                </>
              )}
            </div>

            <div className="form-group">
              <label>Target Language</label>
              <select 
                className="select-input" 
                value={lang} 
                onChange={(e) => setLang(e.target.value)}
              >
                <option value="Spanish">Spanish</option>
                <option value="English">English</option>
                <option value="French">French</option>
                <option value="German">German</option>
              </select>
            </div>

            <div className="form-group">
              <label>Voice Generation Mode</label>
              <div className="voice-mode-cards">
                <div
                  className={`voice-mode-card ${voiceMode === 'clone' ? 'active' : ''}`}
                  onClick={() => setVoiceMode('clone')}
                  role="button"
                  tabIndex={0}
                >
                  <div className="voice-mode-header">
                    <div className="voice-mode-icon-title">
                      <Sparkles size={18} className="voice-mode-icon" />
                      <strong>Clone Actor Voices</strong>
                    </div>
                    <span className="badge-recommend">Zero-Shot AI</span>
                  </div>
                  <p className="voice-mode-desc">
                    Extracts each actor's unique vocal timbre from Demucs isolated vocals and clones it in real-time.
                  </p>
                </div>

                <div
                  className={`voice-mode-card ${voiceMode === 'preset' ? 'active' : ''}`}
                  onClick={() => setVoiceMode('preset')}
                  role="button"
                  tabIndex={0}
                >
                  <div className="voice-mode-header">
                    <div className="voice-mode-icon-title">
                      <Users size={18} className="voice-mode-icon" />
                      <strong>Preset Voice Catalog</strong>
                    </div>
                    <span className="badge-standard">Studio Pool</span>
                  </div>
                  <p className="voice-mode-desc">
                    Selects curated demographic studio voices matched to detected age and gender pools.
                  </p>
                </div>
              </div>
            </div>

            <button 
              className="btn-primary" 
              onClick={handleUpload} 
              disabled={!file || isUploading}
            >
              {isUploading ? (
                <><Loader2 className="spinner" size={20} /> Processing...</>
              ) : (
                'Start Translation'
              )}
            </button>
            
            {error && <div className="error-message">{error}</div>}
          </>
        )}

        {jobId && !job && (
          <div className="job-status" style={{ textAlign: 'center', padding: '3rem 0' }}>
            <Loader2 className="spinner" size={36} style={{ margin: '0 auto 1rem' }} />
            <h3>Initializing Pipeline...</h3>
            <p style={{ color: 'var(--text-secondary)' }}>Connecting to orchestrator and starting extraction.</p>
          </div>
        )}

        {job && (
          <div className="job-status">
            <div className="status-header">
              <div>
                <h3>Job Tracking</h3>
                <span className="job-meta-tag">
                  Target: <strong>{job.target_lang}</strong> • Mode: <strong>{job.voice_mode === 'preset' ? '🎙️ Preset Voice Catalog' : '🎭 Actor Voice Cloning'}</strong>
                </span>
              </div>
              <span className={`status-badge status-${job.status}`}>
                {job.status}
              </span>
            </div>

            <div className="stage-list">
              {displayStages.map((stage) => {
                const isCompleted = stage.status === 'completed';
                const isActive = stage.status === 'processing';
                const isFailed = stage.status === 'failed';
                const isPending = stage.status === 'pending';
                
                return (
                  <div key={stage.key} className={`stage-item ${isCompleted ? 'completed' : isActive ? 'active' : isFailed ? 'failed' : ''}`}>
                    <div className="stage-icon">
                      {isCompleted ? <CheckCircle2 size={18} /> : 
                       isActive ? <Loader2 className="spinner" size={18} /> : 
                       <stage.Icon size={18} />}
                    </div>
                    <div className="stage-info">
                      <h4>{stage.label}</h4>
                      <p>
                        {isCompleted ? 'Done' : 
                         isActive ? 'Processing...' : 
                         isFailed ? (stage.error || 'Failed') :
                         isPending ? 'Waiting in queue' : stage.status}
                      </p>
                    </div>
                  </div>
                );
              })}
            </div>
            
            {Object.keys(emotionsSummary).length > 0 && (
              <div className="emotions-panel">
                <div className="emotions-panel-title">
                  <Smile size={16} style={{ color: 'var(--accent-color)' }} />
                  <span>Speech Emotion Recognition (Whisper-Large-v3)</span>
                </div>
                <div className="emotions-chips">
                  {Object.entries(emotionsSummary).map(([emotion, count]) => {
                    const cfg = EMOTION_MAP[emotion] || { label: emotion, emoji: '🎙️', color: '#94a3b8', bg: 'rgba(255,255,255,0.05)' };
                    return (
                      <span
                        key={emotion}
                        className="emotion-badge"
                        style={{ color: cfg.color, backgroundColor: cfg.bg, borderColor: cfg.color }}
                      >
                        <span>{cfg.emoji}</span>
                        <span>{cfg.label}: {count}</span>
                      </span>
                    );
                  })}
                </div>
              </div>
            )}

            {segments.length > 0 && (
              <div style={{ marginTop: '1rem' }}>
                <button
                  type="button"
                  onClick={() => setShowTimeline(!showTimeline)}
                  style={{
                    background: 'transparent',
                    border: 'none',
                    color: 'var(--text-secondary)',
                    cursor: 'pointer',
                    fontSize: '0.8rem',
                    display: 'flex',
                    alignItems: 'center',
                    gap: '0.4rem',
                    padding: 0
                  }}
                >
                  {showTimeline ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
                  <span>{showTimeline ? 'Hide' : 'Show'} Dialogue & Emotion Timeline ({segments.length} segments)</span>
                </button>

                {showTimeline && (
                  <div className="segments-timeline">
                    {segments.map((seg, idx) => {
                      const emo = seg.emotion || 'neutral';
                      const cfg = EMOTION_MAP[emo] || { label: emo, emoji: '🎙️', color: '#94a3b8', bg: 'rgba(255,255,255,0.05)' };
                      return (
                        <div key={idx} className="segment-card">
                          <div className="segment-header">
                            <span className="segment-speaker">{seg.speaker_label}</span>
                            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                              <span
                                className="emotion-badge"
                                style={{
                                  fontSize: '0.65rem',
                                  padding: '0.1rem 0.4rem',
                                  color: cfg.color,
                                  backgroundColor: cfg.bg,
                                  borderColor: cfg.color
                                }}
                              >
                                {cfg.emoji} {cfg.label}
                                {seg.emotion_confidence ? ` ${(seg.emotion_confidence * 100).toFixed(0)}%` : ''}
                              </span>
                              <span className="segment-time">
                                {formatTimeMs(seg.start_ms)} - {formatTimeMs(seg.end_ms)}
                              </span>
                            </div>
                          </div>
                          <div className="segment-text">{seg.source_text}</div>
                          {seg.translated_text && (
                            <div className="segment-subtext">↳ {seg.translated_text}</div>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {job.status === 'completed' && segmentCount > 0 && (
              <div className="speakers-summary-box">
                <div className="speakers-summary-header">
                  <span>Dubbing Summary ({segmentCount} segments)</span>
                  <span className="speakers-badge">
                    {job.voice_mode === 'preset' ? '🎙️ Preset Voices' : '✨ Zero-Shot Cloned'}
                  </span>
                </div>
                {speakers.length > 0 && (
                  <div className="speakers-grid">
                    {speakers.map(s => {
                      const isCloned = s.fish_reference_id === 'cloned_inline' || s.voice_mode === 'clone';
                      return (
                        <div key={s.speaker_label} className="speaker-pill">
                          <span className="speaker-label">{s.speaker_label}</span>
                          <span className="speaker-arrow">→</span>
                          {isCloned ? (
                            <span className="speaker-clone-tag">
                              <Sparkles size={13} /> Cloned Actor Vocals
                            </span>
                          ) : (
                            <span className="speaker-preset-tag">
                              Voice: <code>{s.fish_reference_id.slice(0, 10)}…</code>
                            </span>
                          )}
                        </div>
                      );
                    })}
                  </div>
                )}
              </div>
            )}

            {syncReport && (
              <div className="speakers-summary-box">
                <div className="speakers-summary-header">
                  <span style={{ display: 'flex', alignItems: 'center', gap: '0.45rem' }}>
                    <BarChart3 size={17} /> Synchronization Quality
                  </span>
                  <span className="speakers-badge">{syncReport.score}/100 · {syncReport.grade}</span>
                </div>
                <div className="speakers-grid">
                  <div className="speaker-pill">Max drift: {syncReport.summary.max_start_drift_ms} ms</div>
                  <div className="speaker-pill">Regenerated: {syncReport.summary.regenerated_segments}</div>
                  <div className="speaker-pill">Compressed: {syncReport.summary.compressed_segments}</div>
                  <div className="speaker-pill">Clipped: {syncReport.summary.clipped_segments}</div>
                </div>
                <a
                  href={`${API_URL}/jobs/${jobId}/sync-report`}
                  download
                  className="btn-primary"
                  style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.45rem', textDecoration: 'none', marginTop: '1rem' }}
                >
                  <Download size={16} /> Download Synchronization Report
                </a>
              </div>
            )}

            {job.status === 'completed' && finalVideoUrl && (
              <div style={{ marginTop: '2rem' }}>
                <h4 style={{ marginBottom: '0.8rem', color: 'var(--text-primary)' }}>Dubbed Result</h4>
                <video controls src={finalVideoUrl} style={{ width: '100%', borderRadius: '12px', border: '1px solid var(--border-color)', background: '#000' }} />
                <a
                  href={finalVideoUrl}
                  download
                  className="btn-primary"
                  style={{ display: 'inline-block', textAlign: 'center', textDecoration: 'none', marginTop: '1rem', width: '100%', boxSizing: 'border-box' }}
                >
                  Download Dubbed Video
                </a>
              </div>
            )}

            {(job.status === 'completed' || job.status === 'failed') && (
              <button className="btn-primary" style={{ marginTop: '1.5rem', background: 'transparent', border: '1px solid var(--border-color)' }} onClick={() => {
                setJobId(null);
                setJob(null);
                setFile(null);
                setSpeakers([]);
                setSegmentCount(0);
                setEmotionsSummary({});
                setSegments([]);
                setSyncReport(null);
                setShowTimeline(false);
              }}>
                Dub Another Video
              </button>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

export default App;
