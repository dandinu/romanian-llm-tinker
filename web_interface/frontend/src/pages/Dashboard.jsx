import React, { useEffect, useState } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Badge } from '@/components/ui/badge'
import { trainingAPI, datasetAPI, checkpointsAPI, healthCheck } from '@/lib/api'
import { Activity, Database, HardDrive, Cpu, TrendingUp } from 'lucide-react'
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts'

export function Dashboard() {
  const [stats, setStats] = useState({
    trainingJobs: 0,
    datasets: 0,
    checkpoints: 0,
    systemStatus: 'loading'
  })
  const [recentJobs, setRecentJobs] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    loadDashboardData()
  }, [])

  const loadDashboardData = async () => {
    try {
      setLoading(true)

      const [healthRes, jobsRes, datasetsRes, checkpointsRes] = await Promise.all([
        healthCheck(),
        trainingAPI.listJobs().catch(() => ({ data: [] })),
        datasetAPI.list().catch(() => ({ data: [] })),
        checkpointsAPI.list().catch(() => ({ data: { checkpoints: [] } }))
      ])

      setStats({
        trainingJobs: jobsRes.data?.length || 0,
        datasets: datasetsRes.data?.length || 0,
        checkpoints: checkpointsRes.data?.checkpoints?.length || 0,
        systemStatus: healthRes.data?.status || 'unknown'
      })

      setRecentJobs(jobsRes.data?.slice(0, 5) || [])
    } catch (error) {
      console.error('Failed to load dashboard data:', error)
    } finally {
      setLoading(false)
    }
  }

  // Mock data for the chart
  const chartData = [
    { step: 0, loss: 4.2 },
    { step: 100, loss: 3.8 },
    { step: 200, loss: 3.2 },
    { step: 300, loss: 2.7 },
    { step: 400, loss: 2.3 },
    { step: 500, loss: 2.0 },
    { step: 600, loss: 1.8 },
    { step: 700, loss: 1.6 },
    { step: 800, loss: 1.5 },
    { step: 900, loss: 1.4 },
    { step: 1000, loss: 1.3 },
  ]

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
    <div className="p-8 space-y-8">
      {/* Header */}
      <div>
        <h1 className="text-3xl font-bold">Dashboard</h1>
        <p className="text-muted-foreground mt-1">
          Monitor your Romanian LLM fine-tuning pipeline
        </p>
      </div>

      {/* Stats Grid */}
      <div className="grid gap-4 md:grid-cols-2 lg:grid-cols-4">
        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Training Jobs</CardTitle>
            <Activity className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.trainingJobs}</div>
            <p className="text-xs text-muted-foreground">Total jobs created</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Datasets</CardTitle>
            <Database className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.datasets}</div>
            <p className="text-xs text-muted-foreground">Available datasets</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">Checkpoints</CardTitle>
            <HardDrive className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold">{stats.checkpoints}</div>
            <p className="text-xs text-muted-foreground">Saved models</p>
          </CardContent>
        </Card>

        <Card>
          <CardHeader className="flex flex-row items-center justify-between space-y-0 pb-2">
            <CardTitle className="text-sm font-medium">System Status</CardTitle>
            <Cpu className="h-4 w-4 text-muted-foreground" />
          </CardHeader>
          <CardContent>
            <div className="text-2xl font-bold capitalize">{stats.systemStatus}</div>
            <p className="text-xs text-muted-foreground">API health</p>
          </CardContent>
        </Card>
      </div>

      {/* Training Progress Chart */}
      <Card>
        <CardHeader>
          <CardTitle>Training Progress</CardTitle>
          <CardDescription>Loss over training steps (sample data)</CardDescription>
        </CardHeader>
        <CardContent>
          <ResponsiveContainer width="100%" height={300}>
            <LineChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="oklch(92.2% 0 0)" />
              <XAxis
                dataKey="step"
                stroke="oklch(45.8% 0.004 258.3)"
                style={{ fontSize: '12px' }}
              />
              <YAxis
                stroke="oklch(45.8% 0.004 258.3)"
                style={{ fontSize: '12px' }}
              />
              <Tooltip
                contentStyle={{
                  backgroundColor: '#fff',
                  border: '1px solid oklch(92.2% 0 0)',
                  borderRadius: '8px'
                }}
              />
              <Line
                type="monotone"
                dataKey="loss"
                stroke="#0B99FF"
                strokeWidth={2}
                dot={{ fill: '#0B99FF', r: 3 }}
              />
            </LineChart>
          </ResponsiveContainer>
        </CardContent>
      </Card>

      {/* Recent Jobs */}
      <Card>
        <CardHeader>
          <CardTitle>Recent Training Jobs</CardTitle>
          <CardDescription>Latest fine-tuning jobs</CardDescription>
        </CardHeader>
        <CardContent>
          {loading ? (
            <p className="text-sm text-muted-foreground">Loading...</p>
          ) : recentJobs.length === 0 ? (
            <p className="text-sm text-muted-foreground">No training jobs yet</p>
          ) : (
            <div className="space-y-3">
              {recentJobs.map((job) => (
                <div
                  key={job.job_id}
                  className="flex items-center justify-between p-3 border border-border rounded-lg hover:bg-accent transition-colors"
                >
                  <div className="flex-1">
                    <div className="font-medium text-sm">{job.job_id}</div>
                    <div className="text-xs text-muted-foreground mt-1">
                      Created: {new Date(job.created_at).toLocaleString()}
                    </div>
                  </div>
                  <div className="flex items-center gap-3">
                    {job.metrics?.final_loss && (
                      <div className="text-right">
                        <div className="text-xs text-muted-foreground">Final Loss</div>
                        <div className="font-medium text-sm">{job.metrics.final_loss.toFixed(3)}</div>
                      </div>
                    )}
                    {getStatusBadge(job.status)}
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
