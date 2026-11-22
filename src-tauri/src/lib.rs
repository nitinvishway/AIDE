use tauri::Manager;
use tauri::Emitter;
use std::fs;
use std::path::{Path, PathBuf};
use chrono::Local;
use serde::{Deserialize, Serialize};
use base64::Engine;

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct IngestedItem {
    pub id: String,
    pub name: String,
    pub file_type: String, // "image", "pdf", "word", "excel", "text", "other"
    pub path: String,
    pub extracted_text_path: Option<String>,
    pub data_url: Option<String>,
    pub size_bytes: u64,
    pub text_preview: Option<String>,
}

#[derive(Serialize, Deserialize, Clone, Debug)]
pub struct MemoryEntry {
    pub timestamp: String,
    pub name: String,
    pub path: String,
    pub summary: String,
    pub keywords: Vec<String>,
    pub text_preview: Option<String>,
}

fn determine_file_type(extension: &str) -> String {
    let ext = extension.to_lowercase();
    match ext.as_str() {
        "png" | "jpg" | "jpeg" | "gif" | "webp" | "bmp" | "svg" | "ico" | "tiff" => "image".to_string(),
        "pdf" => "pdf".to_string(),
        "doc" | "docx" | "odt" | "rtf" => "word".to_string(),
        "xls" | "xlsx" | "csv" | "tsv" | "ods" => "excel".to_string(),
        "txt" | "md" | "json" | "js" | "ts" | "jsx" | "tsx" | "py" | "rs" | "html" | "css" | "log" | "xml" | "yaml" | "yml" => "text".to_string(),
        _ => "other".to_string(),
    }
}

// Get the data directory for persistent storage
fn get_data_dir() -> PathBuf {
    let data_dir = dirs::data_local_dir()
        .unwrap_or_else(|| PathBuf::from("./data"))
        .join("agentsix");
    
    let _ = fs::create_dir_all(&data_dir);
    data_dir
}

#[cfg(target_os = "windows")]
struct TargetWindowCapture {
    pub title: String,
    pub process_name: String,
    pub width: u32,
    pub height: u32,
    pub rgba_bytes: Vec<u8>,
}

#[cfg(target_os = "windows")]
fn capture_active_window_gdi() -> Result<TargetWindowCapture, String> {
    type HDC = *mut std::ffi::c_void;
    type HBITMAP = *mut std::ffi::c_void;
    type HGDIOBJ = *mut std::ffi::c_void;
    type HWND = *mut std::ffi::c_void;
    type HANDLE = *mut std::ffi::c_void;
    type BOOL = i32;

    #[repr(C)]
    #[derive(Default, Copy, Clone)]
    struct RECT {
        left: i32,
        top: i32,
        right: i32,
        bottom: i32,
    }

    #[repr(C)]
    struct BITMAPINFOHEADER {
        bi_size: u32,
        bi_width: i32,
        bi_height: i32,
        bi_planes: u16,
        bi_bit_count: u16,
        bi_compression: u32,
        bi_size_image: u32,
        bi_x_pels_per_meter: i32,
        bi_y_pels_per_meter: i32,
        bi_clr_used: u32,
        bi_clr_important: u32,
    }

    #[repr(C)]
    struct RGBQUAD {
        rgb_blue: u8,
        rgb_green: u8,
        rgb_red: u8,
        rgb_reserved: u8,
    }

    #[repr(C)]
    struct BITMAPINFO {
        bmi_header: BITMAPINFOHEADER,
        bmi_colors: [RGBQUAD; 1],
    }

    extern "system" {
        fn GetDC(hwnd: HWND) -> HDC;
        fn ReleaseDC(hwnd: HWND, hdc: HDC) -> i32;
        fn GetSystemMetrics(n_index: i32) -> i32;
        fn CreateCompatibleDC(hdc: HDC) -> HDC;
        fn CreateCompatibleBitmap(hdc: HDC, cx: i32, cy: i32) -> HBITMAP;
        fn SelectObject(hdc: HDC, h: HGDIOBJ) -> HGDIOBJ;
        fn DeleteDC(hdc: HDC) -> BOOL;
        fn DeleteObject(ho: HGDIOBJ) -> BOOL;
        fn BitBlt(
            hdc: HDC,
            x: i32,
            y: i32,
            cx: i32,
            cy: i32,
            hdc_src: HDC,
            x1: i32,
            y1: i32,
            rop: u32,
        ) -> BOOL;
        fn GetDIBits(
            hdc: HDC,
            hbm: HBITMAP,
            start: u32,
            c_lines: u32,
            lpv_bits: *mut std::ffi::c_void,
            lpbmi: *mut BITMAPINFO,
            usage: u32,
        ) -> i32;

        fn GetForegroundWindow() -> HWND;
        fn GetWindow(hwnd: HWND, ucmd: u32) -> HWND;
        fn IsWindowVisible(hwnd: HWND) -> BOOL;
        fn IsIconic(hwnd: HWND) -> BOOL;
        fn GetWindowTextW(hwnd: HWND, lpstring: *mut u16, nmaxcount: i32) -> i32;
        fn GetWindowRect(hwnd: HWND, lprect: *mut RECT) -> BOOL;
        fn GetWindowThreadProcessId(hwnd: HWND, lpprocessid: *mut u32) -> u32;
        fn GetCurrentProcessId() -> u32;
        fn OpenProcess(dwdesiredaccess: u32, binherithandle: BOOL, dwprocessid: u32) -> HANDLE;
        fn CloseHandle(hobject: HANDLE) -> BOOL;
    }

    #[link(name = "dwmapi")]
    extern "system" {
        fn DwmGetWindowAttribute(
            hwnd: HWND,
            dwattribute: u32,
            pvattribute: *mut std::ffi::c_void,
            cbattribute: u32,
        ) -> i32;
    }

    #[link(name = "psapi")]
    extern "system" {
        fn GetModuleBaseNameW(
            hprocess: HANDLE,
            hmodule: *mut std::ffi::c_void,
            lpbasename: *mut u16,
            nsize: u32,
        ) -> u32;
    }

    const SM_CXSCREEN: i32 = 0;
    const SM_CYSCREEN: i32 = 1;
    const SRCCOPY: u32 = 0x00CC0020;
    const CAPTUREBLT: u32 = 0x40000000;
    const BI_RGB: u32 = 0;
    const DIB_RGB_COLORS: u32 = 0;
    const GW_HWNDNEXT: u32 = 2;
    const DWMWA_EXTENDED_FRAME_BOUNDS: u32 = 9;
    const PROCESS_QUERY_INFORMATION: u32 = 0x0400;
    const PROCESS_VM_READ: u32 = 0x0010;

    unsafe {
        let my_pid = GetCurrentProcessId();
        let mut curr_hwnd = GetForegroundWindow();
        let mut target_hwnd = std::ptr::null_mut();
        let mut target_title = String::new();
        let mut target_process = String::new();
        let mut target_rect = RECT::default();

        // Search for top visible active application window that is not our own agent widget
        let mut checks = 0;
        while !curr_hwnd.is_null() && checks < 50 {
            checks += 1;
            let mut pid: u32 = 0;
            GetWindowThreadProcessId(curr_hwnd, &mut pid);

            let visible = IsWindowVisible(curr_hwnd) != 0;
            let minimized = IsIconic(curr_hwnd) != 0;

            if pid != my_pid && visible && !minimized {
                let mut title_buf = [0u16; 512];
                let len = GetWindowTextW(curr_hwnd, title_buf.as_mut_ptr(), 512);
                if len > 0 {
                    let title = String::from_utf16_lossy(&title_buf[..len as usize])
                        .trim()
                        .to_string();
                    // Skip shell / system overlay windows if empty or taskbar
                    if !title.is_empty()
                        && title != "Program Manager"
                        && title != "Task Switching"
                        && title != "Windows Input Experience"
                    {
                        let mut rect = RECT::default();
                        let dwm_res = DwmGetWindowAttribute(
                            curr_hwnd,
                            DWMWA_EXTENDED_FRAME_BOUNDS,
                            &mut rect as *mut _ as *mut std::ffi::c_void,
                            std::mem::size_of::<RECT>() as u32,
                        );
                        if dwm_res != 0 {
                            GetWindowRect(curr_hwnd, &mut rect);
                        }

                        let w = rect.right - rect.left;
                        let h = rect.bottom - rect.top;
                        if w > 100 && h > 100 {
                            // Extract process name (e.g. chrome.exe, code.exe)
                            let h_proc = OpenProcess(
                                PROCESS_QUERY_INFORMATION | PROCESS_VM_READ,
                                0,
                                pid,
                            );
                            let mut proc_name = String::new();
                            if !h_proc.is_null() {
                                let mut name_buf = [0u16; 260];
                                let nlen = GetModuleBaseNameW(
                                    h_proc,
                                    std::ptr::null_mut(),
                                    name_buf.as_mut_ptr(),
                                    260,
                                );
                                if nlen > 0 {
                                    proc_name = String::from_utf16_lossy(&name_buf[..nlen as usize])
                                        .trim()
                                        .to_string();
                                }
                                CloseHandle(h_proc);
                            }

                            target_hwnd = curr_hwnd;
                            target_title = title;
                            target_process = proc_name;
                            target_rect = rect;
                            break;
                        }
                    }
                }
            }
            curr_hwnd = GetWindow(curr_hwnd, GW_HWNDNEXT);
        }

        // Determine capture area
        let (capture_x, capture_y, capture_w, capture_h) = if !target_hwnd.is_null() {
            let scr_w = GetSystemMetrics(SM_CXSCREEN);
            let scr_h = GetSystemMetrics(SM_CYSCREEN);
            let left = target_rect.left.max(0);
            let top = target_rect.top.max(0);
            let right = target_rect.right.min(scr_w);
            let bottom = target_rect.bottom.min(scr_h);
            let w = (right - left).max(1);
            let h = (bottom - top).max(1);
            (left, top, w, h)
        } else {
            let w = GetSystemMetrics(SM_CXSCREEN).max(1);
            let h = GetSystemMetrics(SM_CYSCREEN).max(1);
            (0, 0, w, h)
        };

        let hdc_screen = GetDC(std::ptr::null_mut());
        if hdc_screen.is_null() {
            return Err("Failed to get desktop device context".to_string());
        }

        let hdc_mem = CreateCompatibleDC(hdc_screen);
        if hdc_mem.is_null() {
            ReleaseDC(std::ptr::null_mut(), hdc_screen);
            return Err("Failed to create memory DC".to_string());
        }

        let hbitmap = CreateCompatibleBitmap(hdc_screen, capture_w, capture_h);
        if hbitmap.is_null() {
            DeleteDC(hdc_mem);
            ReleaseDC(std::ptr::null_mut(), hdc_screen);
            return Err("Failed to create compatible bitmap".to_string());
        }

        let h_old = SelectObject(hdc_mem, hbitmap);

        let blt_res = BitBlt(
            hdc_mem,
            0,
            0,
            capture_w,
            capture_h,
            hdc_screen,
            capture_x,
            capture_y,
            SRCCOPY | CAPTUREBLT,
        );

        if blt_res == 0 {
            SelectObject(hdc_mem, h_old);
            DeleteObject(hbitmap);
            DeleteDC(hdc_mem);
            ReleaseDC(std::ptr::null_mut(), hdc_screen);
            return Err("BitBlt screen capture failed".to_string());
        }

        let mut bi = BITMAPINFO {
            bmi_header: BITMAPINFOHEADER {
                bi_size: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                bi_width: capture_w,
                bi_height: -capture_h, // negative for top-down DIB
                bi_planes: 1,
                bi_bit_count: 32,
                bi_compression: BI_RGB,
                bi_size_image: (capture_w * capture_h * 4) as u32,
                bi_x_pels_per_meter: 0,
                bi_y_pels_per_meter: 0,
                bi_clr_used: 0,
                bi_clr_important: 0,
            },
            bmi_colors: [RGBQUAD {
                rgb_blue: 0,
                rgb_green: 0,
                rgb_red: 0,
                rgb_reserved: 0,
            }],
        };

        let mut raw_bytes = vec![0u8; (capture_w * capture_h * 4) as usize];
        let lines_copied = GetDIBits(
            hdc_mem,
            hbitmap,
            0,
            capture_h as u32,
            raw_bytes.as_mut_ptr() as *mut std::ffi::c_void,
            &mut bi,
            DIB_RGB_COLORS,
        );

        SelectObject(hdc_mem, h_old);
        DeleteObject(hbitmap);
        DeleteDC(hdc_mem);
        ReleaseDC(std::ptr::null_mut(), hdc_screen);

        if lines_copied == 0 {
            return Err("GetDIBits failed to copy screen bitmap".to_string());
        }

        // Convert BGRA to RGBA
        let mut rgba_bytes = vec![0u8; (capture_w * capture_h * 4) as usize];
        for i in 0..(capture_w * capture_h) as usize {
            let b = raw_bytes[i * 4];
            let g = raw_bytes[i * 4 + 1];
            let r = raw_bytes[i * 4 + 2];
            rgba_bytes[i * 4] = r;
            rgba_bytes[i * 4 + 1] = g;
            rgba_bytes[i * 4 + 2] = b;
            rgba_bytes[i * 4 + 3] = 255;
        }

        Ok(TargetWindowCapture {
            title: target_title,
            process_name: target_process,
            width: capture_w as u32,
            height: capture_h as u32,
            rgba_bytes,
        })
    }
}

#[cfg(target_os = "windows")]
fn capture_screen_gdi() -> Result<(u32, u32, Vec<u8>), String> {
    match capture_active_window_gdi() {
        Ok(cap) => Ok((cap.width, cap.height, cap.rgba_bytes)),
        Err(e) => Err(e),
    }
}

#[cfg(not(target_os = "windows"))]
fn capture_screen_gdi() -> Result<(u32, u32, Vec<u8>), String> {
    Err("Direct screen capture is only supported on Windows".to_string())
}

// Live Screen Capture (Direct Windows Desktop & Active Target Window Capture)
#[tauri::command]
fn capture_live_screen() -> Result<IngestedItem, String> {
    #[cfg(target_os = "windows")]
    {
        if let Ok(cap) = capture_active_window_gdi() {
            let data_dir = get_data_dir();
            let timestamp = Local::now().format("%Y%m%d_%H%M%S_%3f").to_string();
            let filename = format!("screen_capture_{}.png", timestamp);
            let file_path = data_dir.join(&filename);

            if let Some(img) = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(
                cap.width,
                cap.height,
                cap.rgba_bytes,
            ) {
                if img.save(&file_path).is_ok() {
                    if let Ok(image_bytes) = fs::read(&file_path) {
                        let engine = base64::engine::general_purpose::STANDARD;
                        let data_url =
                            format!("data:image/png;base64,{}", engine.encode(&image_bytes));

                        let display_name =
                            if !cap.process_name.is_empty() && !cap.title.is_empty() {
                                format!("{} — {}", cap.process_name, cap.title)
                            } else if !cap.title.is_empty() {
                                cap.title.clone()
                            } else {
                                "Live Screen Capture".to_string()
                            };

                        let text_preview = if !cap.title.is_empty() {
                            Some(format!(
                                "Active Application: {} | Window: \"{}\"",
                                cap.process_name, cap.title
                            ))
                        } else {
                            Some("Live Desktop Screen Capture".to_string())
                        };

                        return Ok(IngestedItem {
                            id: timestamp,
                            name: display_name,
                            file_type: "image".to_string(),
                            path: file_path.to_string_lossy().to_string(),
                            extracted_text_path: None,
                            data_url: Some(data_url),
                            size_bytes: image_bytes.len() as u64,
                            text_preview,
                        });
                    }
                }
            }
        }
    }

    let (width, height, rgba_bytes) = capture_screen_gdi()
        .or_else(|_err| -> Result<(u32, u32, Vec<u8>), String> {
            let mut clipboard =
                arboard::Clipboard::new().map_err(|e| format!("Clipboard error: {}", e))?;
            let img = clipboard.get_image().map_err(|e| {
                format!("Direct screen capture & clipboard fallback failed: {}", e)
            })?;
            Ok((img.width as u32, img.height as u32, img.bytes.to_vec()))
        })?;

    let data_dir = get_data_dir();
    let timestamp = Local::now().format("%Y%m%d_%H%M%S_%3f").to_string();
    let filename = format!("screen_capture_{}.png", timestamp);
    let file_path = data_dir.join(&filename);

    let img = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(
        width,
        height,
        rgba_bytes,
    )
    .ok_or("Failed to create image from screen pixels")?;

    img.save(&file_path)
        .map_err(|e| format!("Failed to save captured screen: {}", e))?;

    let image_bytes =
        fs::read(&file_path).map_err(|e| format!("Failed to read captured image: {}", e))?;
    let engine = base64::engine::general_purpose::STANDARD;
    let data_url = format!("data:image/png;base64,{}", engine.encode(&image_bytes));

    Ok(IngestedItem {
        id: timestamp,
        name: "Live Screen Capture".to_string(),
        file_type: "image".to_string(),
        path: file_path.to_string_lossy().to_string(),
        extracted_text_path: None,
        data_url: Some(data_url),
        size_bytes: image_bytes.len() as u64,
        text_preview: Some("Live Desktop Screen Capture".to_string()),
    })
}

// Save image from clipboard
#[tauri::command]
fn capture_screenshot() -> Result<IngestedItem, String> {
    let mut clipboard = arboard::Clipboard::new()
        .map_err(|e| format!("Failed to access clipboard: {}", e))?;
    
    let image = clipboard.get_image()
        .map_err(|e| format!("No image in clipboard: {}", e))?;
    
    let data_dir = get_data_dir();
    let timestamp = Local::now().format("%Y%m%d_%H%M%S_%3f").to_string();
    let filename = format!("screenshot_{}.png", timestamp);
    let file_path = data_dir.join(&filename);
    
    // Convert arboard image to image crate format
    let img = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(
        image.width as u32,
        image.height as u32,
        image.bytes.to_vec(),
    ).ok_or("Failed to create image")?;
    
    img.save(&file_path)
        .map_err(|e| format!("Failed to save image: {}", e))?;
    
    let image_bytes = fs::read(&file_path)
        .map_err(|e| format!("Failed to read saved image: {}", e))?;
    let engine = base64::engine::general_purpose::STANDARD;
    let data_url = format!("data:image/png;base64,{}", engine.encode(&image_bytes));
    
    Ok(IngestedItem {
        id: timestamp,
        name: filename,
        file_type: "image".to_string(),
        path: file_path.to_string_lossy().to_string(),
        extracted_text_path: None,
        data_url: Some(data_url),
        size_bytes: image_bytes.len() as u64,
        text_preview: None,
    })
}

fn extract_pdf_safe(path: &Path, max_chars: usize) -> Option<String> {
    // Strategy 1: lopdf document page extraction (safe & robust across complex PDFs)
    if let Ok(bytes) = fs::read(path) {
        let lopdf_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            if let Ok(doc) = lopdf::Document::load_mem(&bytes) {
                let pages = doc.get_pages();
                let mut full_text = String::new();
                for (page_num, _) in pages {
                    if let Ok(page_text) = doc.extract_text(&[page_num]) {
                        full_text.push_str(&page_text);
                        full_text.push(' ');
                        if full_text.len() >= max_chars {
                            break;
                        }
                    }
                }
                let cleaned: String = full_text.split_whitespace().collect::<Vec<_>>().join(" ");
                if !cleaned.trim().is_empty() {
                    return Some(cleaned.chars().take(max_chars).collect::<String>());
                }
            }
            None
        }));

        if let Ok(Some(text)) = lopdf_res {
            return Some(text);
        }
    }

    // Strategy 2: pdf-extract wrapped in AssertUnwindSafe to prevent any crash/panic
    let extract_res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        pdf_extract::extract_text(path).ok().and_then(|txt| {
            let cleaned: String = txt.split_whitespace().collect::<Vec<_>>().join(" ");
            if cleaned.trim().is_empty() {
                None
            } else {
                Some(cleaned.chars().take(max_chars).collect::<String>())
            }
        })
    }));

    if let Ok(Some(text)) = extract_res {
        return Some(text);
    }

    // Strategy 3: Fallback raw stream text scanner for compressed/complex objects
    if let Ok(bytes) = fs::read(path) {
        let mut text = String::new();
        let mut in_parentheses = false;
        let mut current_token = Vec::new();

        for &b in &bytes {
            if b == b'(' && !in_parentheses {
                in_parentheses = true;
                current_token.clear();
            } else if b == b')' && in_parentheses {
                in_parentheses = false;
                if let Ok(s) = std::str::from_utf8(&current_token) {
                    let cleaned = s.trim();
                    if cleaned.len() > 1 && cleaned.chars().all(|c| c.is_ascii_graphic() || c.is_ascii_whitespace()) {
                        text.push_str(cleaned);
                        text.push(' ');
                    }
                }
                if text.len() >= max_chars {
                    break;
                }
            } else if in_parentheses {
                current_token.push(b);
            }
        }

        let cleaned: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if cleaned.len() > 20 {
            return Some(cleaned.chars().take(max_chars).collect());
        }
    }

    None
}

fn extract_document_text(path: &Path, max_chars: usize) -> Option<String> {
    if !path.exists() {
        return None;
    }
    let ext = path.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
    match ext.as_str() {
        "pdf" => extract_pdf_safe(path, max_chars),
        "txt" | "md" | "json" | "csv" | "tsv" | "xml" | "yaml" | "yml" | "log" | "rs" | "js" | "ts" | "py" | "html" | "css" | "java" | "c" | "cpp" | "h" | "sh" | "bat" | "ps1" => {
            fs::read_to_string(path).ok().and_then(|txt| {
                if txt.trim().is_empty() {
                    None
                } else {
                    Some(txt.chars().take(max_chars).collect())
                }
            })
        }
        _ => None,
    }
}

// Ingest a file from drag-drop
#[tauri::command]
fn ingest_file(path: String) -> Result<IngestedItem, String> {
    let source_path = PathBuf::from(&path);
    
    if !source_path.exists() {
        return Err("File does not exist".to_string());
    }
    
    let data_dir = get_data_dir();
    let file_name = source_path.file_name()
        .ok_or("Invalid file name")?
        .to_string_lossy()
        .to_string();
    
    let ext = source_path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_string();
    
    let file_type = determine_file_type(&ext);
    
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let dest_filename = format!("{}_{}", timestamp, file_name);
    let dest_path = data_dir.join(&dest_filename);
    
    fs::copy(&source_path, &dest_path)
        .map_err(|e| format!("Failed to copy file: {}", e))?;
    
    let metadata = fs::metadata(&dest_path).ok();
    let size_bytes = metadata.map(|m| m.len()).unwrap_or(0);
    
    let mut data_url = None;
    let mut text_preview = None;
    let mut extracted_text_path = None;
    
    if file_type == "image" {
        if let Ok(bytes) = fs::read(&dest_path) {
            let engine = base64::engine::general_purpose::STANDARD;
            let mime = match ext.to_lowercase().as_str() {
                "jpg" | "jpeg" => "image/jpeg",
                "png" => "image/png",
                "gif" => "image/gif",
                "webp" => "image/webp",
                "svg" => "image/svg+xml",
                _ => "image/png",
            };
            data_url = Some(format!("data:{};base64,{}", mime, engine.encode(&bytes)));
        }
    } else {
        // Extract full text and write to a temporary .txt file
        if let Some(full_text) = extract_document_text(&dest_path, 80000) {
            let txt_name = format!("temp_extracted_{}_{}.txt", timestamp, file_name);
            let txt_path = data_dir.join(&txt_name);
            if fs::write(&txt_path, &full_text).is_ok() {
                extracted_text_path = Some(txt_path.to_string_lossy().to_string());
            }
            text_preview = Some(full_text.chars().take(250).collect());
        }
    }
    
    Ok(IngestedItem {
        id: timestamp,
        name: file_name,
        file_type,
        path: dest_path.to_string_lossy().to_string(),
        extracted_text_path,
        data_url,
        size_bytes,
        text_preview,
    })
}

// Ingest file from raw Base64 bytes (HTML5 FileReader fallback)
#[tauri::command]
fn ingest_file_bytes(name: String, base64_data: String) -> Result<IngestedItem, String> {
    let engine = base64::engine::general_purpose::STANDARD;
    let bytes = engine.decode(&base64_data)
        .map_err(|e| format!("Failed to decode base64: {}", e))?;
    
    let data_dir = get_data_dir();
    let ext = Path::new(&name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_string();
    let file_type = determine_file_type(&ext);
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let dest_filename = format!("{}_{}", timestamp, name);
    let dest_path = data_dir.join(&dest_filename);

    fs::write(&dest_path, &bytes)
        .map_err(|e| format!("Failed to write file: {}", e))?;

    let size_bytes = bytes.len() as u64;
    let mut data_url = None;
    let mut text_preview = None;
    let mut extracted_text_path = None;

    if file_type == "image" {
        let mime = match ext.to_lowercase().as_str() {
            "jpg" | "jpeg" => "image/jpeg",
            "png" => "image/png",
            "gif" => "image/gif",
            "webp" => "image/webp",
            "svg" => "image/svg+xml",
            _ => "image/png",
        };
        data_url = Some(format!("data:{};base64,{}", mime, base64_data));
    } else {
        // Extract full text and write to a temporary .txt file
        if let Some(full_text) = extract_document_text(&dest_path, 80000) {
            let txt_name = format!("temp_extracted_{}_{}.txt", timestamp, name);
            let txt_path = data_dir.join(&txt_name);
            if fs::write(&txt_path, &full_text).is_ok() {
                extracted_text_path = Some(txt_path.to_string_lossy().to_string());
            }
            text_preview = Some(full_text.chars().take(250).collect());
        }
    }

    Ok(IngestedItem {
        id: timestamp,
        name,
        file_type,
        path: dest_path.to_string_lossy().to_string(),
        extracted_text_path,
        data_url,
        size_bytes,
        text_preview,
    })
}

// Ingest text from drag-drop
#[tauri::command]
fn ingest_text(text: String) -> Result<IngestedItem, String> {
    let data_dir = get_data_dir();
    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
    let filename = format!("temp_extracted_{}_snippet.txt", timestamp);
    let file_path = data_dir.join(&filename);
    
    fs::write(&file_path, &text)
        .map_err(|e| format!("Failed to save text: {}", e))?;
    
    let preview: String = text.chars().take(250).collect();
    let size_bytes = text.len() as u64;
    
    Ok(IngestedItem {
        id: timestamp,
        name: "Text Snippet".to_string(),
        file_type: "text".to_string(),
        path: file_path.to_string_lossy().to_string(),
        extracted_text_path: Some(file_path.to_string_lossy().to_string()),
        data_url: None,
        size_bytes,
        text_preview: Some(preview),
    })
}

// Delete file
#[tauri::command]
fn delete_file(path: String) -> Result<bool, String> {
    let p = Path::new(&path);
    if p.exists() {
        let _ = fs::remove_file(p);
    }
    Ok(true)
}

// Share item (copy path/content to clipboard)
#[tauri::command]
fn share_item(path: String) -> Result<String, String> {
    let mut clipboard = arboard::Clipboard::new()
        .map_err(|e| format!("Clipboard error: {}", e))?;
    
    let p = Path::new(&path);
    if p.exists() {
        clipboard.set_text(path.clone())
            .map_err(|e| format!("Failed to set clipboard: {}", e))?;
        Ok(format!("Copied file path to clipboard: {}", path))
    } else {
        Err("File does not exist".to_string())
    }
}

// Open file in default application
#[tauri::command]
fn open_file_in_default_app(path: String) -> Result<String, String> {
    let norm_path = path.replace('/', "\\");
    let p = Path::new(&norm_path);
    
    let full_path = if p.is_relative() {
        std::env::current_dir()
            .map(|cd| cd.join(p))
            .unwrap_or_else(|_| p.to_path_buf())
    } else {
        p.to_path_buf()
    };

    if !full_path.exists() {
        return Err(format!("File does not exist: {}", norm_path));
    }

    let clean_path_str = full_path.to_string_lossy().to_string();
    let target = clean_path_str
        .strip_prefix(r"\\?\")
        .unwrap_or(&clean_path_str);

    #[cfg(target_os = "windows")]
    {
        use std::os::windows::process::CommandExt;
        const CREATE_NO_WINDOW: u32 = 0x08000000;
        std::process::Command::new("cmd")
            .args(["/C", "start", "", target])
            .creation_flags(CREATE_NO_WINDOW)
            .spawn()
            .map_err(|e| format!("Failed to launch file: {}", e))?;
        Ok(format!("Opened {}", target))
    }
    #[cfg(not(target_os = "windows"))]
    {
        std::process::Command::new("xdg-open")
            .arg(target)
            .spawn()
            .map_err(|e| format!("Failed to launch file: {}", e))?;
        Ok(format!("Opened {}", target))
    }
}

// Simple add to memory
#[tauri::command]
fn add_to_memory(path: String, name: String) -> Result<String, String> {
    let data_dir = get_data_dir();
    let memory_file = data_dir.join("memory_log.jsonl");
    let timestamp = Local::now().to_rfc3339();
    let entry = format!("{{\"timestamp\":\"{}\",\"name\":\"{}\",\"path\":\"{}\"}}\n", timestamp, name, path);
    
    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(memory_file)
        .map_err(|e| format!("Failed to open memory file: {}", e))?;
    
    file.write_all(entry.as_bytes())
        .map_err(|e| format!("Failed to write to memory file: {}", e))?;
    
    Ok("Saved to memory".to_string())
}

fn clean_model_response(raw: &str) -> String {
    let mut text = raw.trim();

    // 1. Locate the beginning of actual generated model content
    if let Some(pos) = text.rfind("<start_of_turn>model") {
        text = &text[pos + "<start_of_turn>model".len()..];
    } else if let Some(pos) = text.rfind("... (truncated)") {
        text = &text[pos + "... (truncated)".len()..];
    } else if let Some(pos) = text.rfind("Assistant:") {
        text = &text[pos + "Assistant:".len()..];
    } else if let Some(pos) = text.rfind("/glob <pattern>") {
        let after_glob = &text[pos + "/glob <pattern>".len()..];
        // Skip prompt echo if present
        if let Some(user_pos) = after_glob.rfind("<start_of_turn>user") {
            let after_user = &after_glob[user_pos..];
            if let Some(model_pos) = after_user.find("<start_of_turn>model") {
                text = &after_user[model_pos + "<start_of_turn>model".len()..];
            } else if let Some(nl) = after_user.rfind("\n\n") {
                text = &after_user[nl + 2..];
            } else {
                text = after_glob;
            }
        } else {
            text = after_glob;
        }
    } else if let Some(pos) = text.rfind("available commands:") {
        let after_cmd = &text[pos..];
        if let Some(nl) = after_cmd.rfind("\n\n") {
            text = &after_cmd[nl + 2..];
        }
    }

    // 2. Strip trailing telemetry, metrics, and interactive prompts
    if let Some(pos) = text.rfind("\nllama_print_timings:") {
        text = &text[..pos];
    }
    if let Some(pos) = text.find("\n[ Prompt:") {
        text = &text[..pos];
    } else if let Some(pos) = text.find("[ Prompt:") {
        text = &text[..pos];
    }
    if let Some(pos) = text.find("\n[ prompt:") {
        text = &text[..pos];
    } else if let Some(pos) = text.find("[ prompt:") {
        text = &text[..pos];
    }
    if let Some(pos) = text.find("\nExiting...") {
        text = &text[..pos];
    } else if let Some(pos) = text.find("Exiting...") {
        text = &text[..pos];
    }
    // Only strip trailing prompt marker if at the very end of output
    let trimmed_end = text.trim_end();
    if trimmed_end.ends_with("\n>") || trimmed_end.ends_with("\n> ") {
        if let Some(pos) = text.rfind("\n>") {
            text = &text[..pos];
        }
    }

    // 3. Remove turn tags and special tokens
    let cleaned = text
        .replace("<start_of_turn>", "")
        .replace("<end_of_turn>", "")
        .replace("<eos>", "")
        .replace("<|im_end|>", "")
        .replace("<|im_start|>", "")
        .replace("<end_of_utterance>", "")
        .replace("<|endoftext|>", "")
        .trim()
        .to_string();

    cleaned
}

// Low-level Gemma 2 runner via llama-cli.exe
async fn run_llama_gemma(
    app: &tauri::AppHandle,
    formatted_prompt: &str,
    max_tokens: u32,
) -> Result<String, String> {
    let resource_dir = app.path().resource_dir().ok();

    let binary_candidates = [
        resource_dir.as_ref().map(|r| r.join("binaries").join("llama-cli.exe")),
        resource_dir.as_ref().map(|r| r.join("binaries").join("llama-cli-x86_64-pc-windows-msvc.exe")),
        Some(PathBuf::from("src-tauri/binaries/llama-cli.exe")),
        Some(PathBuf::from("binaries/llama-cli.exe")),
        Some(PathBuf::from("src-tauri/binaries/llama-mtmd-cli.exe")),
    ];

    let binary_path = binary_candidates
        .into_iter()
        .flatten()
        .find(|p| p.exists())
        .ok_or_else(|| "Binary (llama-cli.exe) not found in src-tauri/binaries/".to_string())?;

    let model_candidates = [
        resource_dir.as_ref().map(|r| r.join("models").join("gemma-2-2b-it-q4_k_m.gguf")),
        Some(PathBuf::from("src-tauri/models/gemma-2-2b-it-q4_k_m.gguf")),
        Some(PathBuf::from("models/gemma-2-2b-it-q4_k_m.gguf")),
        Some(PathBuf::from("src-tauri/models/gemma-2-2b-it.gguf")),
    ];

    let model_path = model_candidates
        .into_iter()
        .flatten()
        .find(|p| p.exists())
        .ok_or_else(|| "Gemma 2 model (gemma-2-2b-it-q4_k_m.gguf) not found in src-tauri/models/".to_string())?;

    let binary = binary_path.clone();
    let model = model_path.clone();
    let p = formatted_prompt.to_string();
    let n_str = max_tokens.to_string();

    let output = tauri::async_runtime::spawn_blocking(move || {
        let temp_dir = get_data_dir();
        let prompt_file = temp_dir.join(format!("temp_gemma_prompt_{}_{}.txt", Local::now().format("%Y%m%d_%H%M%S_%3f"), std::process::id()));
        if let Err(e) = fs::write(&prompt_file, &p) {
            return Err(std::io::Error::new(std::io::ErrorKind::Other, format!("Failed to write prompt file: {e}")));
        }

        let mut cmd = std::process::Command::new(binary);
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        cmd.args([
            "-m", model.to_string_lossy().as_ref(),
            "-f", prompt_file.to_string_lossy().as_ref(),
            "-c", "8192",
            "-t", "12",
            "-n", n_str.as_str(),
            "--temp", "0.3",
            "--repeat-penalty", "1.1",
            "--single-turn",
            "--simple-io",
            "--no-display-prompt",
        ]);

        let res = cmd.output();
        let _ = fs::remove_file(&prompt_file);
        res
    })
    .await
    .map_err(|e| format!("Task execution error: {e}"))?
    .map_err(|e| format!("Failed to run Gemma 2: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();

    if !output.status.success() {
        return Err(format!("Gemma 2 error (code {:?}): {}", output.status.code(), stderr));
    }

    if stdout.is_empty() {
        return Err(format!("No response produced. Details: {stderr}"));
    }

    let clean = clean_model_response(&stdout);
    if clean.is_empty() {
        Ok(stdout)
    } else {
        Ok(clean)
    }
}

// Add to memory with automatic AI keyword extraction & document summary
// Add to memory with automatic AI keyword extraction & document summary
#[tauri::command]
async fn save_to_memory_with_ai(
    app: tauri::AppHandle,
    path: String,
    name: String,
    extracted_text_path: Option<String>,
) -> Result<MemoryEntry, String> {
    // 1. Read document text from extracted_text_path or original file
    let mut doc_text = extracted_text_path
        .as_ref()
        .and_then(|p| fs::read_to_string(p).ok())
        .or_else(|| extract_document_text(Path::new(&path), 8000));

    // If it's an image, run vision analysis to generate visual description
    if doc_text.is_none() {
        let ext = Path::new(&path).extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
        if ["png", "jpg", "jpeg", "gif", "webp", "bmp"].contains(&ext.as_str()) {
            if let Ok(desc) = caption_image(app.clone(), path.clone(), Some("Describe this image in detail: text, UI elements, and main content.".to_string())).await {
                doc_text = Some(format!("Image Visual Analysis:\n{}", desc));
            }
        }
    }

    let final_doc_text = doc_text.unwrap_or_else(|| format!("Document: {}", name));

    // 2. Ask Gemma to extract a 2-sentence summary and 5-8 topical keywords
    let prompt = format!(
        "<start_of_turn>user\nAnalyze this document / image content.\n1. Write a clear, concise 2-sentence summary.\n2. Extract 5 to 8 important keywords/topics (comma-separated).\n\nOutput strictly in this format:\nSUMMARY: <your summary here>\nKEYWORDS: <keyword1>, <keyword2>, <keyword3>, <keyword4>, <keyword5>\n\nContent:\n\"\"\"\n{}\n\"\"\"<end_of_turn>\n<start_of_turn>model\n",
        final_doc_text.chars().take(4000).collect::<String>()
    );

    let gemma_out = run_llama_gemma(&app, &prompt, 512).await.unwrap_or_else(|_| "Summary generated.".to_string());

    // 3. Parse summary & keywords
    let mut summary = String::new();
    let mut keywords = Vec::new();

    for line in gemma_out.lines() {
        let trimmed = line.trim();
        if trimmed.to_uppercase().starts_with("SUMMARY:") {
            summary = trimmed["SUMMARY:".len()..].trim().to_string();
        } else if trimmed.to_uppercase().starts_with("KEYWORDS:") {
            let kw_str = trimmed["KEYWORDS:".len()..].trim();
            keywords = kw_str
                .split(',')
                .map(|k| k.trim().trim_matches(|c| c == '*' || c == '-' || c == '"' || c == '`').to_string())
                .filter(|k| !k.is_empty())
                .collect();
        }
    }

    if summary.is_empty() {
        summary = gemma_out.lines().take(2).collect::<Vec<_>>().join(" ");
    }
    if keywords.is_empty() {
        keywords = vec![name.clone()];
    }

    let timestamp = Local::now().to_rfc3339();
    let text_preview = Some(final_doc_text.chars().take(250).collect());

    let entry = MemoryEntry {
        timestamp,
        name: name.clone(),
        path: path.clone(),
        summary,
        keywords,
        text_preview,
    };

    let data_dir = get_data_dir();
    let memory_file = data_dir.join("memory_log.jsonl");
    let json_line = serde_json::to_string(&entry).map_err(|e| e.to_string())? + "\n";

    use std::io::Write;
    let mut file = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(memory_file)
        .map_err(|e| format!("Failed to open memory file: {}", e))?;

    file.write_all(json_line.as_bytes())
        .map_err(|e| format!("Failed to write to memory file: {}", e))?;

    Ok(entry)
}

// Offline Vision AI / Image Captioning with llama-mtmd-cli & SmolVLM
#[tauri::command]
async fn caption_image(
    app: tauri::AppHandle,
    image_path: String,
    prompt: Option<String>,
) -> Result<String, String> {
    let default_prompt = "Describe this image in detail, including all visible text, UI components, code, graphics, titles, and layout.".to_string();
    let user_prompt = prompt.unwrap_or(default_prompt);

    let resource_dir = app.path().resource_dir().ok();

    // 1. Locate binary (supports llama-mtmd-cli, llama-cli, llama-minicpmv-cli)
    let binary_candidates = [
        resource_dir.as_ref().map(|r| r.join("binaries").join("llama-mtmd-cli.exe")),
        resource_dir.as_ref().map(|r| r.join("binaries").join("llama-cli.exe")),
        resource_dir.as_ref().map(|r| r.join("binaries").join("llama-mtmd-cli-x86_64-pc-windows-msvc.exe")),
        Some(PathBuf::from("src-tauri/binaries/llama-mtmd-cli.exe")),
        Some(PathBuf::from("src-tauri/binaries/llama-cli.exe")),
        Some(PathBuf::from("src-tauri/binaries/llama-minicpmv-cli.exe")),
        Some(PathBuf::from("src-tauri/binaries/llama-mtmd-cli-x86_64-pc-windows-msvc.exe")),
        Some(PathBuf::from("binaries/llama-mtmd-cli.exe")),
        Some(PathBuf::from("binaries/llama-cli.exe")),
    ];

    let binary_path = binary_candidates
        .into_iter()
        .flatten()
        .find(|p| p.exists())
        .ok_or_else(|| "Binary (llama-mtmd-cli.exe or llama-cli.exe) not found.\nPlace it in src-tauri/binaries/".to_string())?;

    // 2. Locate model
    let model_candidates = [
        resource_dir.as_ref().map(|r| r.join("models").join("smolvlm-256m-q4_k_m.gguf")),
        Some(PathBuf::from("src-tauri/models/smolvlm-256m-q4_k_m.gguf")),
        Some(PathBuf::from("models/smolvlm-256m-q4_k_m.gguf")),
        Some(PathBuf::from("src-tauri/models/model.gguf")),
    ];

    let model_path = model_candidates
        .into_iter()
        .flatten()
        .find(|p| p.exists())
        .ok_or_else(|| "SmolVLM model (.gguf) not found.\nPlace smolvlm-256m-q4_k_m.gguf in src-tauri/models/".to_string())?;

    // 3. Locate vision projector
    let mmproj_candidates = [
        resource_dir.as_ref().map(|r| r.join("models").join("mmproj-smolvlm-256m-f16.gguf")),
        Some(PathBuf::from("src-tauri/models/mmproj-smolvlm-256m-f16.gguf")),
        Some(PathBuf::from("models/mmproj-smolvlm-256m-f16.gguf")),
        Some(PathBuf::from("src-tauri/models/mmproj.gguf")),
    ];

    let mmproj_path = mmproj_candidates
        .into_iter()
        .flatten()
        .find(|p| p.exists())
        .ok_or_else(|| "Vision projector (mmproj-*.gguf) not found.\nPlace mmproj-smolvlm-256m-f16.gguf in src-tauri/models/".to_string())?;

    // 4. Spawn using std::process::Command on worker thread
    let binary = binary_path.clone();
    let model = model_path.clone();
    let mmproj = mmproj_path.clone();
    let img = image_path.clone();
    let p = user_prompt.clone();

    let output = tauri::async_runtime::spawn_blocking(move || {
        let mut cmd = std::process::Command::new(binary);
        #[cfg(target_os = "windows")]
        {
            use std::os::windows::process::CommandExt;
            const CREATE_NO_WINDOW: u32 = 0x08000000;
            cmd.creation_flags(CREATE_NO_WINDOW);
        }
        cmd.args([
            "-m", model.to_string_lossy().as_ref(),
            "--mmproj", mmproj.to_string_lossy().as_ref(),
            "--image", img.as_str(),
            "-p", p.as_str(),
            "-t", "12",
            "-n", "128",
            "--temp", "0.2",
        ]);

        cmd.output()
    })
    .await
    .map_err(|e| format!("Task execution error: {e}"))?
    .map_err(|e| format!("Failed to run sidecar: {e}"))?;

    let stdout = String::from_utf8_lossy(&output.stdout).trim().to_string();
    let stderr = String::from_utf8_lossy(&output.stderr).trim().to_string();

    if !output.status.success() {
        return Err(format!("Engine error (code {:?}): {}", output.status.code(), stderr));
    }

    let clean = clean_model_response(&stdout);
    if clean.is_empty() {
        Ok(stdout)
    } else {
        Ok(clean)
    }
}

// Offline General Text & Document AI with Google Gemma 2 (2B Instruct)
#[tauri::command]
async fn ask_gemma(
    app: tauri::AppHandle,
    prompt: String,
    context: Option<String>,
    file_path: Option<String>,
    extracted_text_path: Option<String>,
) -> Result<String, String> {
    // 1. Read document text from extracted_text_path or file_path
    let mut extracted_content = extracted_text_path
        .as_ref()
        .and_then(|p| fs::read_to_string(p).ok())
        .or_else(|| {
            file_path.as_ref().and_then(|p| extract_document_text(Path::new(p), 8000))
        });

    // 2. If it's an image and extracted_content is empty, run vision analysis
    if extracted_content.is_none() {
        if let Some(ref fp) = file_path {
            let p = Path::new(fp);
            let ext = p.extension().and_then(|e| e.to_str()).unwrap_or("").to_lowercase();
            if ["png", "jpg", "jpeg", "gif", "webp", "bmp"].contains(&ext.as_str()) {
                let vision_prompt = "Examine this screen capture or image thoroughly. Transcribe all visible text, identify active windows, websites, applications, forms, input fields, labels, placeholders, checkboxes, tables, buttons, instructions, and status messages.".to_string();
                if let Ok(desc) = caption_image(app.clone(), fp.clone(), Some(vision_prompt)).await {
                    let data_dir = get_data_dir();
                    let timestamp = Local::now().format("%Y%m%d_%H%M%S").to_string();
                    let caption_txt_path = data_dir.join(format!("temp_extracted_{}_caption.txt", timestamp));
                    let _ = fs::write(&caption_txt_path, &desc);
                    extracted_content = Some(format!("Screen Visual Content & UI Analysis:\n{}", desc));
                }
            }
        }
    }

    let mut context_sections = Vec::new();
    if let Some(ext) = extracted_content {
        if !ext.trim().is_empty() {
            context_sections.push(format!("Document / Screen Content:\n\"\"\"\n{}\n\"\"\"", ext.trim()));
        }
    }
    if let Some(ctx) = context {
        if !ctx.trim().is_empty() {
            context_sections.push(format!("Dialogue History / Context:\n{}", ctx.trim()));
        }
    }

    let formatted_prompt = if !context_sections.is_empty() {
        format!(
            "<start_of_turn>user\n{}\n\nQuestion / Request: {}\nInstructions: Analyze the content. If explaining a screen or form, describe what is happening, enumerate any form fields or required details to fill, and give clear actionable steps.<end_of_turn>\n<start_of_turn>model\n",
            context_sections.join("\n\n"),
            prompt.trim()
        )
    } else {
        format!(
            "<start_of_turn>user\n{}\nAnswer clearly and concisely.<end_of_turn>\n<start_of_turn>model\n",
            prompt.trim()
        )
    };

    run_llama_gemma(&app, &formatted_prompt, 2048).await
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_autostart::Builder::new().build())
        .plugin(tauri_plugin_opener::init())
        .setup(|app| {
            let window = app.get_webview_window("main").unwrap();
            let main_window = window.clone();

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

            // Monitor clipboard for screenshots
            std::thread::spawn(move || {
                let mut last_image_hash = 0u64;
                loop {
                    std::thread::sleep(std::time::Duration::from_millis(500));
                    
                    if let Ok(mut clipboard) = arboard::Clipboard::new() {
                        if let Ok(image) = clipboard.get_image() {
                            let hash = image.bytes.len() as u64;
                            if hash != last_image_hash {
                                last_image_hash = hash;
                                
                                let data_dir = get_data_dir();
                                let timestamp = Local::now().format("%Y%m%d_%H%M%S_%3f").to_string();
                                let filename = format!("screenshot_{}.png", timestamp);
                                let file_path = data_dir.join(&filename);
                                
                                if let Some(img) = image::ImageBuffer::<image::Rgba<u8>, Vec<u8>>::from_raw(
                                    image.width as u32,
                                    image.height as u32,
                                    image.bytes.to_vec(),
                                ) {
                                    if img.save(&file_path).is_ok() {
                                        if let Ok(image_bytes) = fs::read(&file_path) {
                                            let engine = base64::engine::general_purpose::STANDARD;
                                            let data_url = format!("data:image/png;base64,{}", engine.encode(&image_bytes));
                                            let payload = IngestedItem {
                                                id: timestamp,
                                                name: filename,
                                                file_type: "image".to_string(),
                                                path: file_path.to_string_lossy().to_string(),
                                                extracted_text_path: None,
                                                data_url: Some(data_url),
                                                size_bytes: image_bytes.len() as u64,
                                                text_preview: None,
                                            };
                                            let _ = main_window.emit("new-clipboard-image", payload);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            });

            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            capture_screenshot,
            capture_live_screen,
            ingest_file,
            ingest_file_bytes,
            ingest_text,
            delete_file,
            share_item,
            open_file_in_default_app,
            add_to_memory,
            save_to_memory_with_ai,
            caption_image,
            ask_gemma
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}

