import { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import { Upload, FileVideo, Loader2, CheckCircle2, Mic, TextSelect, FileAudio, Film } from 'lucide-react';
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
}

interface Speaker {
  speaker_label: string;
  fish_reference_id: string;
  target_lang: string;
}

function App() {
  const [file, setFile] = useState<File | null>(null);
  const [lang, setLang] = useState('Spanish');
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [jobId, setJobId] = useState<string | null>(null);
  const [job, setJob] = useState<Job | null>(null);
  const [stages, setStages] = useState<Stage[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [segmentCount, setSegmentCount] = useState(0);

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

    const formData = new FormData();
    formData.append('video', file);
    formData.append('targetLang', lang);

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
      } catch (_) {}
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
              <h3>Job Tracking</h3>
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
            
            {job.status === 'completed' && segmentCount > 0 && (
              <div style={{ marginTop: '1.5rem', fontSize: '0.85rem', color: 'var(--text-secondary)' }}>
                <p style={{ marginBottom: '0.5rem' }}>{segmentCount} segments dubbed.</p>
                {speakers.length > 0 && (
                  <ul style={{ listStyle: 'none', padding: 0, margin: 0 }}>
                    {speakers.map(s => (
                      <li key={s.speaker_label}>
                        {s.speaker_label} → <code>{s.fish_reference_id.slice(0, 12)}…</code>
                      </li>
                    ))}
                  </ul>
                )}
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
