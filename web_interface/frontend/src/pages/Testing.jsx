import React, { useState, useEffect, useRef } from 'react'
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import { testAPI } from '@/lib/api'
import { Send, Loader2, Sparkles } from 'lucide-react'

export function Testing() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [examples, setExamples] = useState([])
  const messagesEndRef = useRef(null)

  useEffect(() => {
    loadExamples()
  }, [])

  useEffect(() => {
    scrollToBottom()
  }, [messages])

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }

  const loadExamples = async () => {
    try {
      const response = await testAPI.getExamples()
      setExamples(response.data.examples || [])
    } catch (error) {
      console.error('Failed to load examples:', error)
    }
  }

  const handleSend = async (promptText) => {
    const prompt = promptText || input
    if (!prompt.trim()) return

    setLoading(true)
    const userMessage = { role: 'user', content: prompt }
    setMessages(prev => [...prev, userMessage])
    setInput('')

    try {
      const response = await testAPI.testPrompt({
        prompt,
        max_length: 200,
        temperature: 0.7,
        top_p: 0.9
      })

      const assistantMessage = {
        role: 'assistant',
        content: response.data.response,
        generation_time: response.data.generation_time
      }

      setMessages(prev => [...prev, assistantMessage])
    } catch (error) {
      console.error('Failed to get response:', error)
      const errorMessage = {
        role: 'assistant',
        content: 'Sorry, there was an error processing your request.',
        error: true
      }
      setMessages(prev => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  return (
    <div className="p-8 h-full flex flex-col">
      <div className="mb-6">
        <h1 className="text-3xl font-bold">Model Testing</h1>
        <p className="text-muted-foreground mt-1">
          Test your fine-tuned model with Romanian prompts
        </p>
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-4 gap-6 min-h-0">
        {/* Chat Interface */}
        <Card className="lg:col-span-3 flex flex-col">
          <CardHeader>
            <CardTitle>Interactive Chat</CardTitle>
            <CardDescription>Send prompts and see model responses</CardDescription>
          </CardHeader>
          <CardContent className="flex-1 flex flex-col min-h-0">
            {/* Messages */}
            <div className="flex-1 overflow-y-auto mb-4 space-y-4 min-h-0">
              {messages.length === 0 ? (
                <div className="flex items-center justify-center h-full text-muted-foreground">
                  <div className="text-center">
                    <Sparkles size={48} className="mx-auto mb-4 opacity-20" />
                    <p>Start a conversation with the model</p>
                    <p className="text-sm mt-1">Try one of the example prompts</p>
                  </div>
                </div>
              ) : (
                <>
                  {messages.map((msg, idx) => (
                    <div
                      key={idx}
                      className={`flex ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
                    >
                      <div
                        className={`max-w-[80%] rounded-lg px-4 py-3 ${
                          msg.role === 'user'
                            ? 'bg-primary text-primary-foreground'
                            : msg.error
                            ? 'bg-destructive/10 text-destructive border border-destructive'
                            : 'bg-muted text-foreground'
                        }`}
                      >
                        <div className="text-sm whitespace-pre-wrap">{msg.content}</div>
                        {msg.generation_time && (
                          <div className="text-xs opacity-70 mt-2">
                            Generated in {msg.generation_time.toFixed(2)}s
                          </div>
                        )}
                      </div>
                    </div>
                  ))}
                  {loading && (
                    <div className="flex justify-start">
                      <div className="bg-muted rounded-lg px-4 py-3">
                        <Loader2 size={16} className="animate-spin" />
                      </div>
                    </div>
                  )}
                  <div ref={messagesEndRef} />
                </>
              )}
            </div>

            {/* Input */}
            <div className="flex gap-2">
              <Input
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Scrie o întrebare în română..."
                disabled={loading}
                className="flex-1"
              />
              <Button onClick={() => handleSend()} disabled={loading || !input.trim()}>
                {loading ? (
                  <Loader2 size={16} className="animate-spin" />
                ) : (
                  <Send size={16} />
                )}
              </Button>
            </div>
          </CardContent>
        </Card>

        {/* Examples Sidebar */}
        <Card className="lg:col-span-1">
          <CardHeader>
            <CardTitle className="text-lg">Example Prompts</CardTitle>
            <CardDescription className="text-xs">Click to try</CardDescription>
          </CardHeader>
          <CardContent>
            <div className="space-y-2">
              {examples.map((example, idx) => (
                <button
                  key={idx}
                  onClick={() => handleSend(example)}
                  disabled={loading}
                  className="w-full text-left p-3 text-sm border border-border rounded-lg hover:bg-accent hover:border-primary transition-colors disabled:opacity-50"
                >
                  {example}
                </button>
              ))}

              {examples.length === 0 && (
                <div className="text-sm text-muted-foreground text-center py-4">
                  Loading examples...
                </div>
              )}
            </div>

            <div className="mt-6 p-3 bg-muted rounded-lg">
              <div className="text-xs font-medium mb-2">Settings</div>
              <div className="space-y-1 text-xs text-muted-foreground">
                <div>Max Length: 200 tokens</div>
                <div>Temperature: 0.7</div>
                <div>Top P: 0.9</div>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
