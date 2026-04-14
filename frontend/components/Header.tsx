export default function Header() {
  return (
    <header className="bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-4xl mx-auto px-4 py-4 flex items-center justify-between">
        <div className="flex items-center gap-3">
          <div className="w-9 h-9 bg-blue-600 rounded-lg flex items-center justify-center">
            <span className="text-white font-bold text-sm">BT</span>
          </div>
          <div>
            <h1 className="text-lg font-semibold text-gray-900">Brain Tumor AI</h1>
            <p className="text-xs text-gray-500">VGG16 MRI Classification</p>
          </div>
        </div>
        <div className="flex items-center gap-2">
          <span className="w-2 h-2 bg-green-500 rounded-full"></span>
          <span className="text-xs text-gray-500">Model Online</span>
        </div>
      </div>
    </header>
  )
}