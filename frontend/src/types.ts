export type Locale = 'es' | 'en';

export type JobStatus = 'pending' | 'queued' | 'processing' | 'completed' | 'failed';

export interface User {
  id: string;
  email: string;
  name?: string | null;
}

export interface Project {
  id: string;
  name: string;
  createdAt?: string;
  updatedAt?: string;
}

export interface JobSummary {
  id: string;
  projectId?: string;
  status: JobStatus;
  targetLang?: string;
  voiceMode?: string;
  originalFilename?: string;
  createdAt?: string;
  updatedAt?: string;
  error?: string | null;
}

export interface Stage {
  id?: string;
  stage: string;
  status: string;
  started_at?: string | null;
  finished_at?: string | null;
  error?: string | null;
  download_url?: string | null;
}

export interface Speaker {
  speaker_label: string;
  fish_reference_id?: string;
  target_lang?: string;
  voice_mode?: string;
}

export interface SegmentItem {
  speaker_label: string;
  start_ms: number;
  end_ms: number;
  source_text: string;
  translated_text?: string;
  emotion?: string;
  emotion_confidence?: number;
}

export interface SyncReport {
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

export interface JobDetails {
  job: JobSummary;
  stages: Stage[];
  speakers: Speaker[];
  segmentCount: number;
  emotionsSummary: Record<string, number>;
  segments: SegmentItem[];
  syncReport: SyncReport | null;
  downloadUrl?: string | null;
}

export interface AuthSession {
  user: User;
}
