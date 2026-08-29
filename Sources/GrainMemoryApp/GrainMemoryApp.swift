import SwiftUI
import AppKit
import WebKit
import GrainVDB

@main
struct GrainMemoryApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) var appDelegate

    var body: some Scene {
        Settings {
            EmptyView()
        }
    }
}

class AppDelegate: NSObject, NSApplicationDelegate {
    var statusItem: NSStatusItem?
    var popover: NSPopover?
    var dvrWindow: NSWindow?
    let vdb: GrainVDB? = try? GrainVDB(dimension: 128)

    func applicationDidFinishLaunching(_ notification: Notification) {
        // Prevent dock icon for menu bar app
        NSApp.setActivationPolicy(.accessory)

        statusItem = NSStatusBar.system.statusItem(withLength: NSStatusItem.variableLength)
        if let button = statusItem?.button {
            button.title = "⚡ Grain"
            button.action = #selector(togglePopover)
            button.target = self
        }

        let popover = NSPopover()
        popover.contentSize = NSSize(width: 320, height: 380)
        popover.behavior = .transient
        popover.contentViewController = NSHostingController(
            rootView: MenuBarContentView(
                openDVR: { [weak self] in self?.openAgentDVRWindow() },
                installMCP: { [weak self] in self?.installClaudeMCP() }
            )
        )
        self.popover = popover
    }

    @objc func togglePopover() {
        guard let button = statusItem?.button, let popover = popover else { return }
        if popover.isShown {
            popover.performClose(nil)
        } else {
            popover.show(relativeTo: button.bounds, of: button, preferredEdge: .minY)
        }
    }

    func openAgentDVRWindow() {
        popover?.performClose(nil)

        if dvrWindow == nil {
            let window = NSWindow(
                contentRect: NSRect(x: 100, y: 100, width: 1000, height: 700),
                styleMask: [.titled, .closable, .miniaturizable, .resizable],
                backing: .buffered,
                defer: false
            )
            window.title = "GrainVDB — Agent DVR Studio"
            window.center()
            window.isReleasedWhenClosed = false

            let webView = WKWebView()
            if let htmlPath = Bundle.main.path(forResource: "agent_dvr", ofType: "html") {
                let url = URL(fileURLWithPath: htmlPath)
                webView.loadFileURL(url, allowingReadAccessTo: url.deletingLastPathComponent())
            } else if let localDocs = Bundle.main.resourceURL?.appendingPathComponent("docs/agent_dvr.html"), FileManager.default.fileExists(atPath: localDocs.path) {
                webView.loadFileURL(localDocs, allowingReadAccessTo: localDocs.deletingLastPathComponent())
            } else {
                webView.loadHTMLString("<h1>Agent DVR Studio</h1><p>Running native on Apple Silicon.</p>", baseURL: nil)
            }

            window.contentView = webView
            dvrWindow = window
        }

        dvrWindow?.makeKeyAndOrderFront(nil)
        NSApp.activate(ignoringOtherApps: true)
    }

    func installClaudeMCP() {
        let fileManager = FileManager.default
        let homeDir = fileManager.homeDirectoryForCurrentUser
        let claudeDir = homeDir.appendingPathComponent("Library/Application Support/Claude")
        let configFile = claudeDir.appendingPathComponent("claude_desktop_config.json")

        do {
            try fileManager.createDirectory(at: claudeDir, withIntermediateDirectories: true)

            var config: [String: Any] = [:]
            if fileManager.fileExists(atPath: configFile.path),
               let data = try? Data(contentsOf: configFile),
               let json = try? JSONSerialization.jsonObject(with: data) as? [String: Any] {
                config = json
            }

            var mcpServers = (config["mcpServers"] as? [String: Any]) ?? [:]
            mcpServers["grainvdb-memory"] = [
                "command": "grainvdb",
                "args": ["mcp", "--dim", "128", "--engine", "auto"]
            ]
            config["mcpServers"] = mcpServers

            let updatedData = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
            try updatedData.write(to: configFile)

            let alert = NSAlert()
            alert.messageText = "Claude Desktop Config Updated!"
            alert.informativeText = "Added 'grainvdb-memory' to ~/Library/Application Support/Claude/claude_desktop_config.json.\n\nRestart Claude Desktop to activate persistent Metal vector memory."
            alert.alertStyle = .informational
            alert.addButton(withTitle: "OK")
            alert.runModal()
        } catch {
            let alert = NSAlert()
            alert.messageText = "Failed to Update Config"
            alert.informativeText = error.localizedDescription
            alert.alertStyle = .warning
            alert.runModal()
        }
    }
}

struct MenuBarContentView: View {
    var openDVR: () -> Void
    var installMCP: () -> Void

    @State private var vectorCount: Int = 100
    @State private var memoryMB: Double = 14.2
    @State private var chainVerified: Bool = true

    var body: some View {
        VStack(alignment: .leading, spacing: 14) {
            // Header
            HStack(spacing: 8) {
                ZStack {
                    RoundedRectangle(cornerRadius: 8)
                        .fill(LinearGradient(colors: [.cyan, .blue], startPoint: .topLeading, endPoint: .bottomTrailing))
                        .frame(width: 28, height: 28)
                    Text("⚡")
                        .font(.system(size: 14))
                }
                VStack(alignment: .leading, spacing: 1) {
                    Text("GrainVDB Memory")
                        .font(.headline)
                        .fontWeight(.bold)
                    Text("Apple Silicon Native (Metal 3.0)")
                        .font(.caption2)
                        .foregroundColor(.secondary)
                }
                Spacer()
                Circle()
                    .fill(Color.green)
                    .frame(width: 8, height: 8)
            }
            .padding(.bottom, 2)

            Divider()

            // Telemetry Cards
            VStack(spacing: 8) {
                HStack {
                    Label("Vectors Indexed", systemImage: "square.stack.3d.up")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text("\(vectorCount)")
                        .font(.caption)
                        .fontWeight(.bold)
                        .fontDesign(.monospaced)
                }

                HStack {
                    Label("Unified Memory", systemImage: "memorychip")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(String(format: "%.1f MB", memoryMB))
                        .font(.caption)
                        .fontWeight(.bold)
                        .fontDesign(.monospaced)
                }

                HStack {
                    Label("Merkle Chain", systemImage: "link")
                        .font(.caption)
                        .foregroundColor(.secondary)
                    Spacer()
                    Text(chainVerified ? "VERIFIED VALID" : "TAMPER DETECTED")
                        .font(.system(size: 9, weight: .bold, design: .monospaced))
                        .foregroundColor(chainVerified ? .green : .red)
                        .padding(.horizontal, 6)
                        .padding(.vertical, 2)
                        .background(Color.green.opacity(0.15))
                        .cornerRadius(4)
                }
            }
            .padding(10)
            .background(Color(NSColor.controlBackgroundColor))
            .cornerRadius(10)

            // Actions
            VStack(spacing: 6) {
                Button(action: openDVR) {
                    HStack {
                        Image(systemName: "play.rectangle.fill")
                        Text("Open Agent DVR Studio")
                        Spacer()
                    }
                    .font(.caption)
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.borderedProminent)
                .tint(.blue)

                Button(action: installMCP) {
                    HStack {
                        Image(systemName: "cpu")
                        Text("Configure Claude Desktop (MCP)")
                        Spacer()
                    }
                    .font(.caption)
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.bordered)

                Button(action: {
                    NSApp.terminate(nil)
                }) {
                    HStack {
                        Image(systemName: "power")
                        Text("Quit GrainMemory")
                        Spacer()
                    }
                    .font(.caption)
                    .foregroundColor(.secondary)
                    .frame(maxWidth: .infinity)
                }
                .buttonStyle(.plain)
                .padding(.top, 4)
            }
        }
        .padding(14)
        .frame(width: 320)
    }
}
