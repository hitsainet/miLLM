function App() {
  return (
    <div className="min-h-screen flex items-center justify-center">
      <div className="card max-w-md">
        <div className="card-header">
          <span className="text-primary-400">miLLM</span> Admin UI
        </div>
        <p className="text-slate-400 mb-4">
          Tailwind CSS is working correctly!
        </p>
        <div className="flex gap-2">
          <span className="badge-success">Success</span>
          <span className="badge-warning">Warning</span>
          <span className="badge-primary">Primary</span>
          <span className="badge-purple">Attached</span>
        </div>
        <div className="mt-4 flex gap-2">
          <button className="btn-primary">Primary</button>
          <button className="btn-secondary">Secondary</button>
        </div>
      </div>
    </div>
  )
}

export default App
