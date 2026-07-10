use tauri::Manager;

// Learn more about Tauri commands at https://tauri.app/develop/calling-rust/
#[tauri::command]
fn greet(name: &str) -> String {
    format!("Hello, {}! You've been greeted from Rust!", name)
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_autostart::Builder::new().build())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let window = app.get_webview_window("main").unwrap();

            // Always-on-top reassertion
            let win_clone = window.clone();
            std::thread::spawn(move || loop {
                std::thread::sleep(std::time::Duration::from_secs(2));
                let _ = win_clone.set_always_on_top(true);
            });

            // Undo Show Desktop minimization
            let win_clone2 = window.clone();
            std::thread::spawn(move || loop {
                std::thread::sleep(std::time::Duration::from_millis(300));
                if let Ok(true) = win_clone2.is_minimized() {
                    let _ = win_clone2.unminimize();
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![greet])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
