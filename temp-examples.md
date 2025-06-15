See the examples below - we don't necessarily want to use 'react', but this should give you a better idea of what we are actually going for. 



<page.tsx> 

```
"use client"

import type React from "react"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Badge } from "@/components/ui/badge"
import { Send, Save, Download, Copy, Play, Code, Eye, Settings, ChevronLeft, ChevronRight } from "lucide-react"

interface Message {
  id: string
  role: "user" | "assistant"
  content: string
  timestamp: Date
  hasArtifact?: boolean
  artifactType?: "code" | "preview" | "document"
}

interface Artifact {
  id: string
  title: string
  type: "react" | "html" | "markdown" | "javascript"
  content: string
  preview?: string
}

export default function AICanvasInterface() {
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "1",
      role: "assistant",
      content:
        "Hello! I can help you create and edit code, documents, and interactive content. What would you like to build today?",
      timestamp: new Date(),
    },
  ])
  const [inputValue, setInputValue] = useState("")
  const [isCanvasOpen, setIsCanvasOpen] = useState(true)
  const [canvasWidth, setCanvasWidth] = useState(50) // percentage
  const [activeArtifact, setActiveArtifact] = useState<Artifact | null>(null)
  const [canvasTab, setCanvasTab] = useState("preview")
  const [isResizing, setIsResizing] = useState(false)
  const resizeRef = useRef<HTMLDivElement>(null)

  // Sample artifacts
  const [artifacts, setArtifacts] = useState<Artifact[]>([
    {
      id: "1",
      title: "React Component",
      type: "react",
      content: `import React, { useState } from 'react'

export default function Counter() {
  const [count, setCount] = useState(0)
  
  return (
    <div className="p-6 max-w-sm mx-auto bg-white rounded-xl shadow-lg">
      <h2 className="text-2xl font-bold text-center mb-4">Counter</h2>
      <div className="text-center">
        <div className="text-4xl font-bold text-blue-600 mb-4">{count}</div>
        <div className="space-x-2">
          <button 
            onClick={() => setCount(count - 1)}
            className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600"
          >
            -
          </button>
          <button 
            onClick={() => setCount(count + 1)}
            className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600"
          >
            +
          </button>
        </div>
      </div>
    </div>
  )
}`,
      preview: "Interactive counter component",
    },
  ])

  useEffect(() => {
    if (artifacts.length > 0 && !activeArtifact) {
      setActiveArtifact(artifacts[0])
    }
  }, [artifacts, activeArtifact])

  const handleSendMessage = () => {
    if (!inputValue.trim()) return

    const newMessage: Message = {
      id: Date.now().toString(),
      role: "user",
      content: inputValue,
      timestamp: new Date(),
    }

    setMessages((prev) => [...prev, newMessage])
    setInputValue("")

    // Simulate AI response
    setTimeout(() => {
      const aiResponse: Message = {
        id: (Date.now() + 1).toString(),
        role: "assistant",
        content: "I've created a new artifact for you. You can see it in the canvas on the right.",
        timestamp: new Date(),
        hasArtifact: true,
        artifactType: "code",
      }
      setMessages((prev) => [...prev, aiResponse])
    }, 1000)
  }

  const handleMouseDown = (e: React.MouseEvent) => {
    setIsResizing(true)
    e.preventDefault()
  }

  useEffect(() => {
    const handleMouseMove = (e: MouseEvent) => {
      if (!isResizing) return

      const containerWidth = window.innerWidth
      const newWidth = (e.clientX / containerWidth) * 100

      if (newWidth >= 30 && newWidth <= 70) {
        setCanvasWidth(100 - newWidth)
      }
    }

    const handleMouseUp = () => {
      setIsResizing(false)
    }

    if (isResizing) {
      document.addEventListener("mousemove", handleMouseMove)
      document.addEventListener("mouseup", handleMouseUp)
    }

    return () => {
      document.removeEventListener("mousemove", handleMouseMove)
      document.removeEventListener("mouseup", handleMouseUp)
    }
  }, [isResizing])

  const renderPreview = () => {
    if (!activeArtifact) return <div className="p-4 text-gray-500">No artifact selected</div>

    if (activeArtifact.type === "react") {
      return (
        <div className="p-4">
          <div className="border rounded-lg p-4 bg-gray-50">
            <div className="p-6 max-w-sm mx-auto bg-white rounded-xl shadow-lg">
              <h2 className="text-2xl font-bold text-center mb-4">Counter</h2>
              <div className="text-center">
                <div className="text-4xl font-bold text-blue-600 mb-4">0</div>
                <div className="space-x-2">
                  <button className="px-4 py-2 bg-red-500 text-white rounded hover:bg-red-600">-</button>
                  <button className="px-4 py-2 bg-green-500 text-white rounded hover:bg-green-600">+</button>
                </div>
              </div>
            </div>
          </div>
        </div>
      )
    }

    return <div className="p-4">Preview not available for this artifact type</div>
  }

  return (
    <div className="h-screen flex bg-gray-50">
      {/* Chat Panel */}
      <div className="flex flex-col bg-white border-r" style={{ width: `${100 - canvasWidth}%` }}>
        {/* Chat Header */}
        <div className="flex items-center justify-between p-4 border-b">
          <div className="flex items-center gap-2">
            <div className="w-8 h-8 bg-green-500 rounded-full flex items-center justify-center">
              <span className="text-white text-sm font-bold">AI</span>
            </div>
            <h1 className="text-lg font-semibold">AI Chat Interface</h1>
          </div>
          <Button variant="ghost" size="sm" onClick={() => setIsCanvasOpen(!isCanvasOpen)}>
            {isCanvasOpen ? <ChevronRight /> : <ChevronLeft />}
          </Button>
        </div>

        {/* Messages */}
        <ScrollArea className="flex-1 p-4">
          <div className="space-y-4">
            {messages.map((message) => (
              <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
                <div
                  className={`max-w-[80%] rounded-lg p-3 ${
                    message.role === "user" ? "bg-blue-500 text-white" : "bg-gray-100 text-gray-900"
                  }`}
                >
                  <p className="text-sm">{message.content}</p>
                  {message.hasArtifact && (
                    <Badge variant="secondary" className="mt-2">
                      <Code className="w-3 h-3 mr-1" />
                      Artifact created
                    </Badge>
                  )}
                </div>
              </div>
            ))}
          </div>
        </ScrollArea>

        {/* Input */}
        <div className="p-4 border-t">
          <div className="flex gap-2">
            <Input
              value={inputValue}
              onChange={(e) => setInputValue(e.target.value)}
              placeholder="Type your message... (Shift+Enter for new line)"
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault()
                  handleSendMessage()
                }
              }}
            />
            <Button onClick={handleSendMessage} size="sm">
              <Send className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </div>

      {/* Resize Handle */}
      {isCanvasOpen && (
        <div
          ref={resizeRef}
          className="w-1 bg-gray-300 cursor-col-resize hover:bg-gray-400 transition-colors"
          onMouseDown={handleMouseDown}
        />
      )}

      {/* Canvas Panel */}
      {isCanvasOpen && (
        <div className="flex flex-col bg-white" style={{ width: `${canvasWidth}%` }}>
          {/* Canvas Header */}
          <div className="flex items-center justify-between p-4 border-b">
            <div className="flex items-center gap-2">
              <h2 className="text-lg font-semibold">Canvas Editor</h2>
              {activeArtifact && <Badge variant="outline">{activeArtifact.type}</Badge>}
            </div>
            <div className="flex items-center gap-2">
              <Button variant="ghost" size="sm">
                <Save className="w-4 h-4" />
              </Button>
              <Button variant="ghost" size="sm">
                <Download className="w-4 h-4" />
              </Button>
              <Button variant="ghost" size="sm">
                <Copy className="w-4 h-4" />
              </Button>
              <Button variant="ghost" size="sm">
                <Settings className="w-4 h-4" />
              </Button>
            </div>
          </div>

          {/* Canvas Tabs */}
          <div className="border-b">
            <Tabs value={canvasTab} onValueChange={setCanvasTab}>
              <TabsList className="w-full justify-start rounded-none border-0 bg-transparent p-0">
                <TabsTrigger
                  value="preview"
                  className="rounded-none border-b-2 border-transparent data-[state=active]:border-blue-500"
                >
                  <Eye className="w-4 h-4 mr-2" />
                  Preview
                </TabsTrigger>
                <TabsTrigger
                  value="code"
                  className="rounded-none border-b-2 border-transparent data-[state=active]:border-blue-500"
                >
                  <Code className="w-4 h-4 mr-2" />
                  Code
                </TabsTrigger>
              </TabsList>
            </Tabs>
          </div>

          {/* Canvas Content */}
          <div className="flex-1 overflow-hidden">
            <Tabs value={canvasTab} className="h-full">
              <TabsContent value="preview" className="h-full m-0">
                <ScrollArea className="h-full">{renderPreview()}</ScrollArea>
              </TabsContent>
              <TabsContent value="code" className="h-full m-0">
                <ScrollArea className="h-full">
                  <div className="p-4">
                    <pre className="bg-gray-900 text-gray-100 p-4 rounded-lg text-sm overflow-x-auto">
                      <code>{activeArtifact?.content || "No code available"}</code>
                    </pre>
                  </div>
                </ScrollArea>
              </TabsContent>
            </Tabs>
          </div>

          {/* Canvas Footer */}
          <div className="p-4 border-t bg-gray-50">
            <div className="flex items-center justify-between text-sm text-gray-600">
              <span>{activeArtifact ? `Editing: ${activeArtifact.title}` : "No artifact selected"}</span>
              <div className="flex items-center gap-2">
                <Button variant="ghost" size="sm">
                  <Play className="w-4 h-4 mr-1" />
                  Run
                </Button>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

```

</page.tsx> 

<scroll-area.tsx>

```
"use client"

import * as React from "react"
import * as ScrollAreaPrimitive from "@radix-ui/react-scroll-area"

import { cn } from "@/lib/utils"

const ScrollArea = React.forwardRef<
  React.ElementRef<typeof ScrollAreaPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof ScrollAreaPrimitive.Root>
>(({ className, children, ...props }, ref) => (
  <ScrollAreaPrimitive.Root ref={ref} className={cn("relative overflow-hidden", className)} {...props}>
    <ScrollAreaPrimitive.Viewport className="h-full w-full rounded-[inherit]">{children}</ScrollAreaPrimitive.Viewport>
    <ScrollBar />
    <ScrollAreaPrimitive.Corner />
  </ScrollAreaPrimitive.Root>
))
ScrollArea.displayName = ScrollAreaPrimitive.Root.displayName

const ScrollBar = React.forwardRef<
  React.ElementRef<typeof ScrollAreaPrimitive.ScrollAreaScrollbar>,
  React.ComponentPropsWithoutRef<typeof ScrollAreaPrimitive.ScrollAreaScrollbar>
>(({ className, orientation = "vertical", ...props }, ref) => (
  <ScrollAreaPrimitive.ScrollAreaScrollbar
    ref={ref}
    orientation={orientation}
    className={cn(
      "flex touch-none select-none transition-colors",
      orientation === "vertical" && "h-full w-2.5 border-l border-l-transparent p-[1px]",
      orientation === "horizontal" && "h-2.5 flex-col border-t border-t-transparent p-[1px]",
      className,
    )}
    {...props}
  >
    <ScrollAreaPrimitive.ScrollAreaThumb className="relative flex-1 rounded-full bg-border" />
  </ScrollAreaPrimitive.ScrollAreaScrollbar>
))
ScrollBar.displayName = ScrollAreaPrimitive.ScrollAreaScrollbar.displayName

export { ScrollArea, ScrollBar }

```

</scroll-area.tsx>

<separator.tsx> 

```
"use client"

import * as React from "react"
import * as SeparatorPrimitive from "@radix-ui/react-separator"

import { cn } from "@/lib/utils"

const Separator = React.forwardRef<
  React.ElementRef<typeof SeparatorPrimitive.Root>,
  React.ComponentPropsWithoutRef<typeof SeparatorPrimitive.Root>
>(({ className, orientation = "horizontal", decorative = true, ...props }, ref) => (
  <SeparatorPrimitive.Root
    ref={ref}
    decorative={decorative}
    orientation={orientation}
    className={cn("shrink-0 bg-border", orientation === "horizontal" ? "h-[1px] w-full" : "h-full w-[1px]", className)}
    {...props}
  />
))
Separator.displayName = SeparatorPrimitive.Root.displayName

export { Separator }

```

</separator.tsx> 