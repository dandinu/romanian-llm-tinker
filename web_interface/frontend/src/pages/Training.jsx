import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { Badge } from '@/components/ui/badge'
import { trainingAPI, datasetAPI } from '@/lib/api'
import { Play, RefreshCw, X, ChevronDown } from 'lucide-react'

export function Training() {
  const [jobs, setJobs] = useState([])
  const [datasets, setDatasets] = useState([])
  const [loading, setLoading] = useState(false)
  const [showNewJobForm, setShowNewJobForm] = useState(false)

  const [formData, setFormData] = useState({
    model_name: 'meta-llama/Llama-3.1-8B',
    max_steps: 1000,
    batch_size: 4,
    learning_rate: 0.0001,
    lora_rank: 8,
    lora_alpha: 16,
    lora_dropout: 0.05,
    warmup_steps: 100,
    output_dir: './checkpoints',
    dataset_path: ''
  })

  useEffect(() => {
    loadJobs()
    loadDatasets()
  }, [])

  const loadJobs = async () => {
    try {
      const response = await trainingAPI.listJobs()
      setJobs(response.data || [])
    } catch (error) {
      console.error('Failed to load jobs:', error)
    }
  }

  const loadDatasets = async () => {
    try {
      const response = await datasetAPI.list()
      setDatasets(response.data || [])
    } catch (error) {
      console.error('Failed to load datasets:', error)
    }
  }

  const handleStartTraining = async (e) => {
    e.preventDefault()
    setLoading(true)

    try {
      await trainingAPI.startTraining(formData)
      setShowNewJobForm(false)
      setFormData({
        ...formData,
        dataset_path: ''
      })
      await loadJobs()
    } catch (error) {
      console.error('Failed to start training:', error)
      alert('Failed to start training job')
    } finally {
      setLoading(false)
    }
  }

  const handleCancelJob = async (jobId) => {
    try {
      await trainingAPI.cancelJob(jobId)
      await loadJobs()
    } catch (error) {
      console.error('Failed to cancel job:', error)
    }
  }

  const getStatusBadge = (status) => {
    const variants = {
      running: 'default',
      completed: 'success',
      failed: 'destructive',
      queued: 'warning',
      cancelled: 'outline'
    }
    return <Badge variant={variants[status] || 'outline'}>{status}</Badge>
  }

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Training</h1>
          <p className="text-muted-foreground mt-1">
            Start and monitor fine-tuning jobs
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={loadJobs}>
            <RefreshCw size={16} className="mr-2" />
            Refresh
          </Button>
          <Button onClick={() => setShowNewJobForm(!showNewJobForm)}>
            <Play size={16} className="mr-2" />
            New Training Job
          </Button>
        </div>
      </div>

      {/* New Job Form */}
      {showNewJobForm && (
        <Card>
          <CardHeader>
            <CardTitle>Configure Training Job</CardTitle>
            <CardDescription>Set parameters for fine-tuning</CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleStartTraining} className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="text-sm font-medium block mb-2">Model Name</label>
                  <Input
                    value={formData.model_name}
                    onChange={(e) => setFormData({ ...formData, model_name: e.target.value })}
                    placeholder="meta-llama/Llama-3.1-8B"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium block mb-2">Dataset</label>
                  <select
                    className="flex h-10 w-full rounded-md border border-input bg-background px-3 py-2 text-sm ring-offset-background focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-primary"
                    value={formData.dataset_path}
                    onChange={(e) => setFormData({ ...formData, dataset_path: e.target.value })}
                    required
                  >
                    <option value="">Select a dataset</option>
                    {datasets.map((ds) => (
                      <option key={ds.path} value={ds.path}>
                        {ds.name} ({ds.num_examples} examples)
                      </option>
                    ))}
                  </select>
                </div>

                <div>
                  <label className="text-sm font-medium block mb-2">Max Steps</label>
                  <Input
                    type="number"
                    value={formData.max_steps}
                    onChange={(e) => setFormData({ ...formData, max_steps: parseInt(e.target.value) })}
                    min="1"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium block mb-2">Batch Size</label>
                  <Input
                    type="number"
                    value={formData.batch_size}
                    onChange={(e) => setFormData({ ...formData, batch_size: parseInt(e.target.value) })}
                    min="1"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium block mb-2">Learning Rate</label>
                  <Input
                    type="number"
                    step="0.00001"
                    value={formData.learning_rate}
                    onChange={(e) => setFormData({ ...formData, learning_rate: parseFloat(e.target.value) })}
                  />
                </div>

                <div>
                  <label className="text-sm font-medium block mb-2">LoRA Rank</label>
                  <Input
                    type="number"
                    value={formData.lora_rank}
                    onChange={(e) => setFormData({ ...formData, lora_rank: parseInt(e.target.value) })}
                    min="1"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium block mb-2">LoRA Alpha</label>
                  <Input
                    type="number"
                    value={formData.lora_alpha}
                    onChange={(e) => setFormData({ ...formData, lora_alpha: parseInt(e.target.value) })}
                    min="1"
                  />
                </div>

                <div>
                  <label className="text-sm font-medium block mb-2">Warmup Steps</label>
                  <Input
                    type="number"
                    value={formData.warmup_steps}
                    onChange={(e) => setFormData({ ...formData, warmup_steps: parseInt(e.target.value) })}
                    min="0"
                  />
                </div>
              </div>

              <div className="flex gap-2 justify-end">
                <Button type="button" variant="outline" onClick={() => setShowNewJobForm(false)}>
                  Cancel
                </Button>
                <Button type="submit" disabled={loading}>
                  {loading ? 'Starting...' : 'Start Training'}
                </Button>
              </div>
            </form>
          </CardContent>
        </Card>
      )}

      {/* Jobs List */}
      <Card>
        <CardHeader>
          <CardTitle>Training Jobs</CardTitle>
          <CardDescription>All training jobs and their status</CardDescription>
        </CardHeader>
        <CardContent>
          {jobs.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Play size={48} className="mx-auto mb-4 opacity-20" />
              <p>No training jobs yet</p>
              <p className="text-sm mt-1">Click "New Training Job" to get started</p>
            </div>
          ) : (
            <div className="space-y-3">
              {jobs.map((job) => (
                <div
                  key={job.job_id}
                  className="border border-border rounded-lg p-4 hover:bg-accent transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <h3 className="font-semibold">{job.job_id}</h3>
                        {getStatusBadge(job.status)}
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 text-sm">
                        <div>
                          <div className="text-muted-foreground">Model</div>
                          <div className="font-medium">{job.config.model_name.split('/')[1]}</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">Steps</div>
                          <div className="font-medium">{job.config.max_steps}</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">Learning Rate</div>
                          <div className="font-medium">{job.config.learning_rate}</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">Created</div>
                          <div className="font-medium">
                            {new Date(job.created_at).toLocaleDateString()}
                          </div>
                        </div>
                      </div>

                      {job.metrics && (
                        <div className="mt-3 p-3 bg-muted rounded-md">
                          <div className="text-sm text-muted-foreground mb-1">Metrics</div>
                          <div className="flex gap-4 text-sm">
                            <span>Final Loss: <strong>{job.metrics.final_loss}</strong></span>
                            <span>Steps: <strong>{job.metrics.steps_completed}</strong></span>
                          </div>
                        </div>
                      )}

                      {job.error && (
                        <div className="mt-3 p-3 bg-destructive/10 text-destructive rounded-md text-sm">
                          Error: {job.error}
                        </div>
                      )}
                    </div>

                    {job.status === 'running' && (
                      <Button
                        variant="destructive"
                        size="sm"
                        onClick={() => handleCancelJob(job.job_id)}
                      >
                        <X size={16} className="mr-1" />
                        Cancel
                      </Button>
                    )}
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>
    </div>
  )
}
