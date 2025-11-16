import React, { useState, useEffect } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Badge } from '@/components/ui/badge'
import { datasetAPI } from '@/lib/api'
import { Upload, RefreshCw, Eye, FileText, Database } from 'lucide-react'

export function Datasets() {
  const [datasets, setDatasets] = useState([])
  const [loading, setLoading] = useState(false)
  const [uploadingFile, setUploadingFile] = useState(false)
  const [previewData, setPreviewData] = useState(null)

  useEffect(() => {
    loadDatasets()
  }, [])

  const loadDatasets = async () => {
    setLoading(true)
    try {
      const response = await datasetAPI.list()
      setDatasets(response.data || [])
    } catch (error) {
      console.error('Failed to load datasets:', error)
    } finally {
      setLoading(false)
    }
  }

  const handleFileUpload = async (event) => {
    const file = event.target.files?.[0]
    if (!file) return

    setUploadingFile(true)
    try {
      await datasetAPI.upload(file)
      await loadDatasets()
      alert('Dataset uploaded successfully!')
    } catch (error) {
      console.error('Failed to upload dataset:', error)
      alert('Failed to upload dataset')
    } finally {
      setUploadingFile(false)
      event.target.value = '' // Reset file input
    }
  }

  const handlePreview = async (datasetName) => {
    try {
      const response = await datasetAPI.preview(datasetName, 5)
      setPreviewData({
        name: datasetName,
        examples: response.data.preview
      })
    } catch (error) {
      console.error('Failed to preview dataset:', error)
      alert('Failed to load preview')
    }
  }

  return (
    <div className="p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold">Datasets</h1>
          <p className="text-muted-foreground mt-1">
            Manage training datasets for Romanian LLM
          </p>
        </div>
        <div className="flex gap-2">
          <Button variant="outline" onClick={loadDatasets} disabled={loading}>
            <RefreshCw size={16} className={`mr-2 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </Button>
          <Button onClick={() => document.getElementById('file-upload').click()} disabled={uploadingFile}>
            <Upload size={16} className="mr-2" />
            {uploadingFile ? 'Uploading...' : 'Upload Dataset'}
          </Button>
          <input
            id="file-upload"
            type="file"
            accept=".jsonl,.json"
            onChange={handleFileUpload}
            className="hidden"
          />
        </div>
      </div>

      {/* Upload Instructions */}
      <Card className="bg-primary/5 border-primary/20">
        <CardContent className="pt-6">
          <div className="flex items-start gap-3">
            <FileText className="text-primary mt-1" size={20} />
            <div>
              <h3 className="font-medium mb-1">Dataset Format</h3>
              <p className="text-sm text-muted-foreground">
                Upload JSONL files with conversation format. Each line should contain:
              </p>
              <pre className="mt-2 p-3 bg-background border border-border rounded-md text-xs overflow-x-auto">
{`{
  "messages": [
    {"role": "user", "content": "Întrebare în română?"},
    {"role": "assistant", "content": "Răspuns în română..."}
  ]
}`}
              </pre>
            </div>
          </div>
        </CardContent>
      </Card>

      {/* Datasets List */}
      <Card>
        <CardHeader>
          <CardTitle>Available Datasets</CardTitle>
          <CardDescription>All uploaded datasets ready for training</CardDescription>
        </CardHeader>
        <CardContent>
          {datasets.length === 0 ? (
            <div className="text-center py-12 text-muted-foreground">
              <Database size={48} className="mx-auto mb-4 opacity-20" />
              <p>No datasets uploaded yet</p>
              <p className="text-sm mt-1">Upload a JSONL file to get started</p>
            </div>
          ) : (
            <div className="space-y-3">
              {datasets.map((dataset) => (
                <div
                  key={dataset.path}
                  className="border border-border rounded-lg p-4 hover:bg-accent transition-colors"
                >
                  <div className="flex items-start justify-between">
                    <div className="flex-1">
                      <div className="flex items-center gap-3 mb-2">
                        <FileText size={20} className="text-primary" />
                        <h3 className="font-semibold">{dataset.name}</h3>
                        <Badge variant="outline">{dataset.num_examples} examples</Badge>
                      </div>

                      <div className="grid grid-cols-2 md:grid-cols-3 gap-4 text-sm mt-3">
                        <div>
                          <div className="text-muted-foreground">Size</div>
                          <div className="font-medium">{dataset.size_mb} MB</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">Examples</div>
                          <div className="font-medium">{dataset.num_examples.toLocaleString()}</div>
                        </div>
                        <div>
                          <div className="text-muted-foreground">Created</div>
                          <div className="font-medium">
                            {new Date(dataset.created_at).toLocaleDateString()}
                          </div>
                        </div>
                      </div>
                    </div>

                    <Button
                      variant="outline"
                      size="sm"
                      onClick={() => handlePreview(dataset.name)}
                    >
                      <Eye size={16} className="mr-1" />
                      Preview
                    </Button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </CardContent>
      </Card>

      {/* Preview Modal */}
      {previewData && (
        <Card className="fixed inset-8 z-50 overflow-auto bg-background shadow-lg">
          <CardHeader>
            <div className="flex items-center justify-between">
              <div>
                <CardTitle>Dataset Preview: {previewData.name}</CardTitle>
                <CardDescription>Showing first 5 examples</CardDescription>
              </div>
              <Button variant="outline" onClick={() => setPreviewData(null)}>
                Close
              </Button>
            </div>
          </CardHeader>
          <CardContent>
            <div className="space-y-4">
              {previewData.examples.map((example, idx) => (
                <div key={idx} className="border border-border rounded-lg p-4">
                  <div className="text-xs font-medium text-muted-foreground mb-2">
                    Example {idx + 1}
                  </div>
                  {example.messages?.map((msg, msgIdx) => (
                    <div key={msgIdx} className="mb-2 last:mb-0">
                      <div className="text-xs font-medium text-primary mb-1">
                        {msg.role}:
                      </div>
                      <div className="text-sm bg-muted p-3 rounded">
                        {msg.content}
                      </div>
                    </div>
                  ))}
                </div>
              ))}
            </div>
          </CardContent>
        </Card>
      )}
    </div>
  )
}
