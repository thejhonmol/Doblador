import axios, { AxiosError } from 'axios';
import type { AuthSession, JobDetails, JobSummary, Project, User } from './types';

const rawBaseUrl = (import.meta.env.VITE_API_BASE_URL as string | undefined) || '/api/v1';
export const API_BASE_URL = rawBaseUrl.replace(/\/$/, '');

const client = axios.create({
  baseURL: API_BASE_URL,
  timeout: 30_000,
  withCredentials: true,
});

function normalizeSession(data: Record<string, unknown>): AuthSession {
  const user = data.user as User | undefined;
  if (!user?.id || !user.email) {
    throw new Error('Invalid authentication response');
  }
  return { user };
}

function normalizeProjects(data: unknown): Project[] {
  if (Array.isArray(data)) return data as Project[];
  if (data && typeof data === 'object' && Array.isArray((data as { projects?: unknown }).projects)) {
    return (data as { projects: Project[] }).projects;
  }
  return [];
}

function normalizeProject(data: unknown): Project {
  if (data && typeof data === 'object' && 'project' in data) {
    return (data as { project: Project }).project;
  }
  return data as Project;
}

export async function login(email: string, password: string): Promise<AuthSession> {
  const { data } = await client.post('/auth/login', { email, password });
  return normalizeSession(data);
}

export async function register(email: string, password: string, inviteCode?: string): Promise<AuthSession> {
  const { data } = await client.post('/auth/register', {
    email,
    password,
    ...(inviteCode?.trim() ? { inviteCode: inviteCode.trim() } : {}),
  });
  return normalizeSession(data);
}

export async function logout(): Promise<void> {
  await client.post('/auth/logout');
}

export async function getCurrentUser(): Promise<User> {
  const { data } = await client.get('/auth/me');
  return (data.user ?? data) as User;
}

export async function listProjects(): Promise<Project[]> {
  const { data } = await client.get('/projects');
  return normalizeProjects(data);
}

export async function createProject(name: string): Promise<Project> {
  const { data } = await client.post('/projects', { name });
  return normalizeProject(data);
}

export async function getProject(projectId: string): Promise<Project> {
  const { data } = await client.get(`/projects/${encodeURIComponent(projectId)}`);
  return normalizeProject(data);
}

export async function listProjectJobs(projectId: string): Promise<JobSummary[]> {
  const { data } = await client.get(`/projects/${encodeURIComponent(projectId)}/jobs`);
  return Array.isArray(data.jobs) ? data.jobs : [];
}

export async function createJob(
  projectId: string,
  file: File,
  targetLang: string,
  voiceMode: 'clone' | 'preset',
): Promise<JobSummary> {
  const formData = new FormData();
  formData.append('video', file);
  formData.append('targetLang', targetLang);
  formData.append('voiceMode', voiceMode);
  const { data } = await client.post(`/projects/${encodeURIComponent(projectId)}/jobs`, formData);
  return data.job as JobSummary;
}

export async function getJob(jobId: string): Promise<JobDetails> {
  const { data } = await client.get(`/jobs/${encodeURIComponent(jobId)}`);
  return normalizeJobDetails(data);
}

function normalizeJobDetails(data: Record<string, unknown>): JobDetails {
  const job = (data.job ?? data) as JobDetails['job'];
  return {
    job,
    stages: (data.stages as JobDetails['stages'] | undefined) ?? [],
    speakers: (data.speakers as JobDetails['speakers'] | undefined) ?? [],
    segmentCount: Number(data.segmentCount ?? data.segment_count ?? 0),
    emotionsSummary: (data.emotionsSummary as Record<string, number> | undefined) ??
      (data.emotions_summary as Record<string, number> | undefined) ?? {},
    segments: (data.segments as JobDetails['segments'] | undefined) ?? [],
    syncReport: (data.syncReport as JobDetails['syncReport'] | undefined) ??
      (data.sync_report as JobDetails['syncReport'] | undefined) ?? null,
    downloadUrl: (data.downloadUrl ?? data.download_url ?? null) as string | null,
  };
}

export function absoluteApiUrl(path: string): string {
  if (/^https?:\/\//i.test(path)) return path;
  const apiRoot = API_BASE_URL.startsWith('http')
    ? new URL(API_BASE_URL).origin
    : window.location.origin;
  return new URL(path, apiRoot).toString();
}

export function jobDownloadUrl(jobId: string): string {
  return `${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}/result`;
}

export function syncReportUrl(jobId: string): string {
  return `${API_BASE_URL}/jobs/${encodeURIComponent(jobId)}/sync-report`;
}

export function apiErrorMessage(error: unknown): string | null {
  if (!axios.isAxiosError(error)) return null;
  const axiosError = error as AxiosError<{ error?: string; message?: string }>;
  return axiosError.response?.data?.message ?? axiosError.response?.data?.error ?? null;
}

export function isUnauthorized(error: unknown): boolean {
  return axios.isAxiosError(error) && error.response?.status === 401;
}

export function isNetworkError(error: unknown): boolean {
  return axios.isAxiosError(error) && !error.response;
}
