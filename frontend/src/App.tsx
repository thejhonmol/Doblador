import { useEffect, useMemo, useRef, useState } from 'react';
import {
  ArrowLeft, CheckCircle2, ChevronDown, ChevronUp, Download, FileVideo,
  FolderOpen, Languages, Loader2, LogOut, Menu, Mic2, Plus, Sparkles,
  Upload, UserRound, X,
} from 'lucide-react';
import {
  absoluteApiUrl, apiErrorMessage, createJob, createProject,
  getCurrentUser, getJob, getProject, isNetworkError,
  isUnauthorized, jobDownloadUrl, listProjectJobs, listProjects, login, logout as apiLogout, register,
  syncReportUrl,
} from './api';
import { translate, type TranslationKey } from './i18n';
import type { AuthSession, JobDetails, JobStatus, Locale, Project, User } from './types';
import './index.css';

type View = { name: 'projects' } | { name: 'project'; projectId: string } | { name: 'new'; projectId?: string };

const LANG_OPTIONS = [
  ['es', 'Español'], ['en', 'English'], ['fr', 'Français'], ['de', 'Deutsch'],
  ['it', 'Italiano'], ['pt', 'Português'], ['ja', '日本語'], ['zh', '中文'],
];
const STAGES = ['extract', 'translate', 'tts', 'assemble'] as const;
const configuredPasswordMinimum = Number(import.meta.env.VITE_MIN_PASSWORD_LENGTH);
const MIN_PASSWORD_LENGTH = Number.isInteger(configuredPasswordMinimum) && configuredPasswordMinimum > 0
  ? configuredPasswordMinimum
  : 12;

function formatDate(value?: string, locale: Locale = 'es') {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.valueOf())) return '—';
  return new Intl.DateTimeFormat(locale === 'es' ? 'es-EC' : 'en-US', { dateStyle: 'medium', timeStyle: 'short' }).format(date);
}

function formatTime(ms: number) {
  const seconds = Math.floor(ms / 1000);
  return `${Math.floor(seconds / 60).toString().padStart(2, '0')}:${(seconds % 60).toString().padStart(2, '0')}`;
}

function App() {
  const [locale, setLocale] = useState<Locale>(() => (localStorage.getItem('zhowplus_locale') === 'en' ? 'en' : 'es'));
  const [user, setUser] = useState<User | null>(null);
  const [booting, setBooting] = useState(true);
  const [view, setView] = useState<View>({ name: 'projects' });
  const [mobileNav, setMobileNav] = useState(false);
  const t = (key: TranslationKey) => translate(locale, key);

  useEffect(() => {
    getCurrentUser().then(setUser).catch(() => setUser(null)).finally(() => setBooting(false));
  }, []);

  const changeLocale = (next: Locale) => {
    localStorage.setItem('zhowplus_locale', next);
    setLocale(next);
  };

  const onSession = (session: AuthSession) => {
    setUser(session.user);
    setView({ name: 'projects' });
  };

  const logout = () => {
    void apiLogout().catch(() => undefined);
    setUser(null);
    setView({ name: 'projects' });
  };

  if (booting) return <FullLoader label={t('loading')} />;
  if (!user) return <AuthView locale={locale} onLocale={changeLocale} onSession={onSession} />;

  return (
    <div className="shell">
      <aside className={`sidebar ${mobileNav ? 'open' : ''}`}>
        <button className="sidebar-close" onClick={() => setMobileNav(false)} aria-label="Close"><X /></button>
        <Brand />
        <nav className="nav-list">
          <button className={view.name === 'projects' ? 'active' : ''} onClick={() => { setView({ name: 'projects' }); setMobileNav(false); }}>
            <FolderOpen size={19} /> {t('projects')}
          </button>
          <button className={view.name === 'new' ? 'active' : ''} onClick={() => { setView({ name: 'new' }); setMobileNav(false); }}>
            <Plus size={19} /> {t('newProject')}
          </button>
        </nav>
        <div className="sidebar-bottom">
          <div className="locale-toggle" aria-label={t('language')}>
            <Languages size={17} />
            <button className={locale === 'es' ? 'active' : ''} onClick={() => changeLocale('es')}>ES</button>
            <button className={locale === 'en' ? 'active' : ''} onClick={() => changeLocale('en')}>EN</button>
          </div>
          <div className="account-row">
            <div className="avatar"><UserRound size={18} /></div>
            <div><strong>{user.name || user.email.split('@')[0]}</strong><span>{user.email}</span></div>
          </div>
          <button className="logout-btn" onClick={logout}><LogOut size={17} /> {t('logout')}</button>
        </div>
      </aside>
      {mobileNav && <button className="sidebar-backdrop" onClick={() => setMobileNav(false)} aria-label="Close menu" />}
      <main className="main-content">
        <header className="mobile-header"><button onClick={() => setMobileNav(true)}><Menu /></button><Brand compact /></header>
        {view.name === 'projects' && <ProjectsView locale={locale} onOpen={(id) => setView({ name: 'project', projectId: id })} onNew={() => setView({ name: 'new' })} />}
        {view.name === 'project' && <ProjectView projectId={view.projectId} locale={locale} onBack={() => setView({ name: 'projects' })} onNew={() => setView({ name: 'new', projectId: view.projectId })} onUnauthorized={logout} />}
        {view.name === 'new' && <NewJobView initialProjectId={view.projectId} locale={locale} onCancel={() => setView({ name: 'projects' })} onCreated={(projectId) => setView({ name: 'project', projectId })} onUnauthorized={logout} />}
      </main>
    </div>
  );
}

function Brand({ compact = false }: { compact?: boolean }) {
  return <div className={`brand ${compact ? 'compact' : ''}`}><span className="brand-mark">Z</span><div><strong>ZhowPlus</strong>{!compact && <span>Studio</span>}</div></div>;
}

function FullLoader({ label }: { label: string }) {
  return <div className="full-loader"><Loader2 className="spin" /><span>{label}</span></div>;
}

function AuthView({ locale, onLocale, onSession }: { locale: Locale; onLocale: (l: Locale) => void; onSession: (s: AuthSession) => void }) {
  const [mode, setMode] = useState<'login' | 'register'>('login');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [inviteCode, setInviteCode] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const t = (key: TranslationKey) => translate(locale, key);

  const submit = async (event: React.FormEvent) => {
    event.preventDefault(); setBusy(true); setError(null);
    try {
      const session = mode === 'login'
        ? await login(email.trim(), password)
        : await register(email.trim(), password, inviteCode);
      onSession(session);
    } catch (err) {
      setError(isNetworkError(err) ? t('networkError') : apiErrorMessage(err) || (mode === 'login' ? t('invalidCredentials') : t('genericError')));
    } finally { setBusy(false); }
  };

  return <div className="auth-page">
    <div className="auth-glow" />
    <div className="auth-top"><Brand /><div className="locale-toggle"><Languages size={17} /><button className={locale === 'es' ? 'active' : ''} onClick={() => onLocale('es')}>ES</button><button className={locale === 'en' ? 'active' : ''} onClick={() => onLocale('en')}>EN</button></div></div>
    <section className="auth-card">
      <div className="eyebrow"><Sparkles size={14} /> ZhowPlus Studio</div>
      <h1>{t(mode === 'login' ? 'loginTitle' : 'registerTitle')}</h1>
      <p>{t(mode === 'login' ? 'loginSubtitle' : 'registerSubtitle')}</p>
      <form onSubmit={submit}>
        <Field label={t('email')} type="email" value={email} onChange={setEmail} autoComplete="email" required />
        <Field
          label={t('password')}
          type="password"
          value={password}
          onChange={setPassword}
          autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
          required
          minLength={mode === 'register' ? MIN_PASSWORD_LENGTH : undefined}
          aria-describedby={mode === 'register' ? 'password-minimum' : undefined}
        />
        {mode === 'register' && <span className="muted" id="password-minimum">{t('passwordMinimum')}: {MIN_PASSWORD_LENGTH}</span>}
        {mode === 'register' && <Field label={t('inviteCode')} value={inviteCode} onChange={setInviteCode} autoComplete="off" placeholder={t('inviteCodeOptional')} />}
        {error && <ErrorBox message={error} />}
        <button className="primary-btn" disabled={busy}>{busy && <Loader2 className="spin" size={18} />}{t(busy ? (mode === 'login' ? 'loggingIn' : 'registering') : mode)}</button>
      </form>
      <div className="auth-switch"><span>{t(mode === 'login' ? 'noAccount' : 'haveAccount')}</span><button onClick={() => { setMode(mode === 'login' ? 'register' : 'login'); setError(null); }}>{t(mode === 'login' ? 'register' : 'login')}</button></div>
    </section>
  </div>;
}

function Field({ label, value, onChange, type = 'text', ...props }: { label: string; value: string; onChange: (v: string) => void; type?: string } & Omit<React.InputHTMLAttributes<HTMLInputElement>, 'value' | 'onChange' | 'type'>) {
  return <label className="field"><span>{label}</span><input type={type} value={value} onChange={(e) => onChange(e.target.value)} {...props} /></label>;
}

function ProjectsView({ locale, onOpen, onNew }: { locale: Locale; onOpen: (id: string) => void; onNew: () => void }) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const t = (key: TranslationKey) => translate(locale, key);
  const load = () => { setLoading(true); setError(null); listProjects().then(setProjects).catch((e) => setError(isNetworkError(e) ? t('networkError') : apiErrorMessage(e) || t('genericError'))).finally(() => setLoading(false)); };
  useEffect(load, []);

  return <div className="page"><PageHeader title={t('projectsTitle')} subtitle={t('projectsSubtitle')} action={<button className="primary-btn small" onClick={onNew}><Plus size={18} />{t('newProject')}</button>} />
    {error ? <ErrorState message={error} retry={load} locale={locale} /> : loading ? <PanelLoader label={t('loading')} /> : projects.length === 0 ? <EmptyProjects locale={locale} onNew={onNew} /> :
      <div className="project-grid">{projects.map((project) => <button className="project-card" key={project.id} onClick={() => onOpen(project.id)}>
        <div className="project-icon"><FileVideo /></div><div className="project-card-main"><h3>{project.name}</h3><p>{t('created')}: {formatDate(project.createdAt, locale)}</p></div>
        <div className="project-card-status"><span className="muted">{t('open')}</span></div>
      </button>)}</div>}
  </div>;
}

function EmptyProjects({ locale, onNew }: { locale: Locale; onNew: () => void }) {
  const t = (key: TranslationKey) => translate(locale, key);
  return <div className="empty-state"><div className="empty-icon"><FolderOpen /></div><h2>{t('emptyProjects')}</h2><p>{t('emptyProjectsHint')}</p><button className="primary-btn small" onClick={onNew}><Plus size={18} />{t('createProject')}</button></div>;
}

function ProjectView({ projectId, locale, onBack, onNew, onUnauthorized }: { projectId: string; locale: Locale; onBack: () => void; onNew: () => void; onUnauthorized: () => void }) {
  const [project, setProject] = useState<Project | null>(null);
  const [selectedJob, setSelectedJob] = useState<JobDetails | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const t = (key: TranslationKey) => translate(locale, key);

  const loadProject = async () => {
    setLoading(true); setError(null);
    try {
      const [data, projectJobs] = await Promise.all([getProject(projectId), listProjectJobs(projectId)]);
      setProject(data);
      const job = projectJobs[0];
      if (job?.id) setSelectedJob(await getJob(job.id)); else setSelectedJob(null);
    } catch (err) {
      if (isUnauthorized(err)) { onUnauthorized(); return; }
      setError(isNetworkError(err) ? t('networkError') : apiErrorMessage(err) || t('genericError'));
    } finally { setLoading(false); }
  };
  useEffect(() => { loadProject(); }, [projectId]);

  useEffect(() => {
    const status = selectedJob?.job.status;
    if (!selectedJob || status === 'completed' || status === 'failed') return;
    const timer = window.setInterval(async () => {
      try {
        const next = await getJob(selectedJob.job.id); setSelectedJob(next);
        if (next.job.status === 'completed' || next.job.status === 'failed') {
          setProject(await getProject(projectId));
        }
      } catch (err) { if (isUnauthorized(err)) onUnauthorized(); }
    }, 3000);
    return () => window.clearInterval(timer);
  }, [selectedJob?.job.id, selectedJob?.job.status, projectId]);

  if (loading) return <div className="page"><PanelLoader label={t('loadingProject')} /></div>;
  if (error || !project) return <div className="page"><ErrorState message={error || t('genericError')} retry={loadProject} locale={locale} /></div>;

  return <div className="page">
    <button className="back-btn" onClick={onBack}><ArrowLeft size={17} />{t('backToProjects')}</button>
    <PageHeader title={project.name} subtitle={`${t('created')}: ${formatDate(project.createdAt, locale)}`} action={<button className="primary-btn small" onClick={onNew}><Plus size={18} />{t('createAnotherJob')}</button>} />
    {selectedJob ? <JobPanel details={selectedJob} locale={locale} /> : <div className="empty-state compact"><h2>{t('noJobs')}</h2><button className="primary-btn small" onClick={onNew}><Upload size={18} />{t('createAnotherJob')}</button></div>}
  </div>;
}

function NewJobView({ initialProjectId, locale, onCancel, onCreated, onUnauthorized }: { initialProjectId?: string; locale: Locale; onCancel: () => void; onCreated: (id: string) => void; onUnauthorized: () => void }) {
  const [projects, setProjects] = useState<Project[]>([]);
  const [projectId, setProjectId] = useState(initialProjectId || '');
  const [projectName, setProjectName] = useState('');
  const [file, setFile] = useState<File | null>(null);
  const [targetLang, setTargetLang] = useState(locale === 'es' ? 'es' : 'en');
  const [voiceMode, setVoiceMode] = useState<'clone' | 'preset'>('clone');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const t = (key: TranslationKey) => translate(locale, key);
  useEffect(() => { listProjects().then((p) => { setProjects(p); if (!projectId && p.length) setProjectId(p[0].id); }).catch(() => undefined); }, []);

  const submit = async () => {
    if (!file) { setError(t('selectVideo')); return; }
    setBusy(true); setError(null);
    try {
      let id = projectId;
      if (!id) {
        if (!projectName.trim()) { setError(t('requiredProjectName')); setBusy(false); return; }
        id = (await createProject(projectName.trim())).id;
      }
      await createJob(id, file, targetLang, voiceMode);
      onCreated(id);
    } catch (err) {
      if (isUnauthorized(err)) { onUnauthorized(); return; }
      setError(isNetworkError(err) ? t('networkError') : apiErrorMessage(err) || t('genericError'));
    } finally { setBusy(false); }
  };

  return <div className="page narrow"><button className="back-btn" onClick={onCancel}><ArrowLeft size={17} />{t('backToProjects')}</button><PageHeader title={t('newProject')} subtitle={t('projectsSubtitle')} />
    <div className="form-card">
      <label className="field"><span>{t('projects')}</span><select value={projectId} onChange={(e) => setProjectId(e.target.value)}><option value="">+ {t('createProject')}</option>{projects.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}</select></label>
      {!projectId && <Field label={t('projectName')} value={projectName} onChange={setProjectName} placeholder={t('projectNamePlaceholder')} />}
      <div className={`upload-zone ${file ? 'selected' : ''}`} role="button" tabIndex={0} onClick={() => inputRef.current?.click()} onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') inputRef.current?.click(); }} onDrop={(e) => { e.preventDefault(); setFile(e.dataTransfer.files?.[0] || null); }} onDragOver={(e) => e.preventDefault()}>
        <input ref={inputRef} type="file" accept="video/*" hidden onChange={(e) => setFile(e.target.files?.[0] || null)} />
        {file ? <><FileVideo size={34} /><strong>{file.name}</strong><span>{(file.size / 1048576).toFixed(1)} MB · {t('fileSelected')}</span></> : <><Upload size={34} /><strong>{t('uploadTitle')}</strong><span>{t('uploadHint')}</span></>}
      </div>
      <div className="form-two"><label className="field"><span>{t('targetLanguage')}</span><select value={targetLang} onChange={(e) => setTargetLang(e.target.value)}>{LANG_OPTIONS.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label></div>
      <div className="field"><span>{t('voiceMode')}</span><div className="choice-grid"><button className={voiceMode === 'clone' ? 'active' : ''} onClick={() => setVoiceMode('clone')}><Sparkles size={20} /><strong>{t('cloneVoice')}</strong><span>{t('cloneVoiceHint')}</span></button><button className={voiceMode === 'preset' ? 'active' : ''} onClick={() => setVoiceMode('preset')}><Mic2 size={20} /><strong>{t('presetVoice')}</strong><span>{t('presetVoiceHint')}</span></button></div></div>
      {error && <ErrorBox message={error} />}
      <button className="primary-btn" onClick={submit} disabled={busy || !file}>{busy && <Loader2 className="spin" size={18} />}{t(busy ? 'creatingJob' : 'startJob')}</button>
    </div>
  </div>;
}

function JobPanel({ details, locale }: { details: JobDetails; locale: Locale }) {
  const [showTimeline, setShowTimeline] = useState(false);
  const t = (key: TranslationKey) => translate(locale, key);
  const stageMap = useMemo(() => new Map(details.stages.map((s) => [s.stage, s])), [details.stages]);
  const download = details.downloadUrl ? absoluteApiUrl(details.downloadUrl) : jobDownloadUrl(details.job.id);

  return <div className="job-layout"><section className="panel"><div className="panel-title"><div><span className="eyebrow">{t('latestJob')}</span><h2>{t('jobProgress')}</h2></div><StatusBadge status={details.job.status} locale={locale} /></div>
    <div className="stage-list">{STAGES.map((name) => { const stage = stageMap.get(name); const status = stage?.status || 'pending'; return <div className={`stage ${status}`} key={name}><div className="stage-dot">{status === 'completed' ? <CheckCircle2 size={18} /> : status === 'processing' ? <Loader2 className="spin" size={18} /> : <span />}</div><div><strong>{t((`stage${name[0].toUpperCase()}${name.slice(1)}`) as TranslationKey)}</strong><p>{stage?.error || t(status === 'completed' ? 'done' : status === 'processing' ? 'working' : status === 'failed' ? 'failed' : 'queued')}</p></div></div>; })}</div>
    {details.job.status === 'failed' && <ErrorBox message={details.job.error || t('genericError')} />}
  </section>
  <section className="panel"><div className="panel-title"><div><span className="eyebrow">ZhowPlus</span><h2>{t('resultTitle')}</h2></div></div>
    {details.job.status === 'completed' ? <><video className="result-video" controls src={download} /><a className="primary-btn" href={download} download><Download size={18} />{t('downloadVideo')}</a></> : <div className="result-placeholder"><Loader2 className={details.job.status === 'failed' ? '' : 'spin'} /><p>{details.job.status === 'failed' ? t('failed') : t('processing')}</p></div>}
  </section>
  {(details.speakers.length > 0 || details.syncReport) && <section className="panel span-two"><div className="metrics-grid">{details.speakers.length > 0 && <div><span>{t('speakers')}</span><strong>{details.speakers.length}</strong></div>}{details.syncReport && <div><span>{t('syncQuality')}</span><strong>{details.syncReport.score}/100 · {details.syncReport.grade}</strong></div>}{details.syncReport && <a href={syncReportUrl(details.job.id)} download>{t('syncQuality')} <Download size={15} /></a>}</div></section>}
  {details.segments.length > 0 && <section className="panel span-two"><button className="timeline-toggle" onClick={() => setShowTimeline(!showTimeline)}>{showTimeline ? <ChevronUp /> : <ChevronDown />} {t(showTimeline ? 'hideTimeline' : 'showTimeline')} ({details.segmentCount || details.segments.length})</button>{showTimeline && <div className="timeline">{details.segments.map((segment, index) => <div className="segment" key={`${segment.start_ms}-${index}`}><div><strong>{segment.speaker_label}</strong><span>{formatTime(segment.start_ms)} – {formatTime(segment.end_ms)}</span></div><p>{segment.source_text}</p>{segment.translated_text && <p className="translated">{segment.translated_text}</p>}</div>)}</div>}</section>}
  </div>;
}

function PageHeader({ title, subtitle, action }: { title: string; subtitle?: string; action?: React.ReactNode }) { return <div className="page-header"><div><h1>{title}</h1>{subtitle && <p>{subtitle}</p>}</div>{action}</div>; }
function PanelLoader({ label }: { label: string }) { return <div className="panel-loader"><Loader2 className="spin" /><span>{label}</span></div>; }
function ErrorBox({ message }: { message: string }) { return <div className="error-box">{message}</div>; }
function ErrorState({ message, retry, locale }: { message: string; retry: () => void; locale: Locale }) { const t = (k: TranslationKey) => translate(locale, k); return <div className="empty-state"><h2>{t('errorTitle')}</h2><p>{message}</p><button className="secondary-btn" onClick={retry}>{t('retry')}</button></div>; }
function StatusBadge({ status, locale }: { status: JobStatus; locale: Locale }) { const key = status === 'queued' ? 'waiting' : status; return <span className={`status-badge ${status}`}>{translate(locale, key as TranslationKey)}</span>; }

export default App;
