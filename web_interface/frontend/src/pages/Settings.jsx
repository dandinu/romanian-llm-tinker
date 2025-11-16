import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { configAPI } from '@/lib/api'
import { Save, RefreshCw } from 'lucide-react'

export function Settings() {
  const [config, setConfig] = useState(null)
  const [loading, setLoading] = useState(false)
  const [saving, setSaving] = useState(false)

  useEffect(() => {
    loadConfig()
  }, [])

  const loadConfig = async () => {
    setLoading(true)
    try {
      const response = await configAPI.get()
      setConfig(response.data)
    } catch (error) {
      console.error('Failed to load config:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleSave = async () => {
    setSaving(true)
    try {
      await configAPI.update(config)
      alert('Configuration saved successfully!')
    } catch (error) {
      console.error('Failed to save config:', error)
      alert('Failed to save configuration')
    } finally {
      setSaving(false)
    }
  }

  if (loading) {
    return (
      <div className="p-8">
        <div className="flex items-center justify-center h-64">
          <RefreshCw className="animate-spin" size={32} />
        </div>
      </div>
    )
  }

  return (
    <div className="p-8 space-y-6">
      <div>
        <h1 className="text-3xl font-bold">Settings</h1>
        <p className="text-muted-foreground mt-1">
          Configure default training parameters
        </p>
      </div>

      <Card>
        <CardHeader>
          <CardTitle>Training Configuration</CardTitle>
          <CardDescription>
            Default parameters for fine-tuning jobs (from hyperparams.yaml)
          </CardDescription>
        </CardHeader>
        <CardContent>
          {config ? (
            <div className="space-y-6">
              <div className="p-4 bg-muted rounded-lg">
                <pre className="text-xs overflow-auto">
                  {JSON.stringify(config, null, 2)}
                </pre>
              </div>

              <div className="flex gap-2 justify-end">
                <Button variant="outline" onClick={loadConfig}>
                  <RefreshCw size={16} className="mr-2" />
                  Reload
                </Button>
                <Button onClick={handleSave} disabled={saving}>
                  <Save size={16} className="mr-2" />
                  {saving ? 'Saving...' : 'Save Configuration'}
                </Button>
              </div>
            </div>
          ) : (
            <p className="text-muted-foreground">No configuration available</p>
          )}
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>About</CardTitle>
          <CardDescription>System information</CardDescription>
        </CardHeader>
        <CardContent className="space-y-2 text-sm">
          <div className="flex justify-between">
            <span className="text-muted-foreground">Version</span>
            <span className="font-medium">1.0.0</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Model</span>
            <span className="font-medium">Llama 3.1 8B</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Framework</span>
            <span className="font-medium">Tinker</span>
          </div>
          <div className="flex justify-between">
            <span className="text-muted-foreground">Language</span>
            <span className="font-medium">Romanian</span>
          </div>
        </CardContent>
      </Card>
    </div>
  )
}
