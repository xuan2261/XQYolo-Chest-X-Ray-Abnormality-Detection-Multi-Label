'''
    @from MangoAI &3836712GKcH2717GhcK. please see https://github.com/iscyy/ultralyticsPro
'''
from ultralytics.utils import IterableSimpleNamespace
from pathlib import Path
import sys
from ultralytics.utils import DEFAULT_CFG
from ultralytics.cfg import get_cfg
import platform
current_os = platform.system()
import yaml
import re
import os

def yaml_load(file="data.yaml", append_filename=False):

    assert Path(file).suffix in (".yaml", ".yml"), f"Attempting to load non-YAML file {file} with yaml_load()"
    with open(file, errors="ignore", encoding="utf-8") as f:
        s = f.read()  # string

        if not s.isprintable():
            s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

        data = yaml.safe_load(s) or {}  
        if append_filename:
            data["yaml_file"] = str(file)
        return data

# def load_script():
#     try:
#         if len(sys.argv) > 2:
#             original_path = sys.argv[2]
#             new_path = original_path.replace("ultralytics\\", "").replace("\\", "/")

#             FILE = Path(__file__).resolve()
#             if current_os == "Windows":
#                 ROOT = FILE.parents[1]
#             elif current_os == "Linux":
#                 ROOT = FILE.parents[2]
#             else:
#                 ROOT = FILE.parents[2]
#             DEFAULT_CFG_PATH = ROOT / new_path
#             DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
#             DEFAULT_CFG_PA = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
#             return DEFAULT_CFG_PA
#         else:
#             # Fallback to default values if no model path is provided
#             return IterableSimpleNamespace(newhead="")
#     except Exception as e:
#         print(f"ERROR in load_script: {e}")
#         # Return empty namespace with newhead attribute to prevent further errors
#         return IterableSimpleNamespace(newhead="")

def load_script():
    print("DEBUG: Starting load_script()")
    print(f"DEBUG: sys.argv = {sys.argv}")
    
    try:
        # Kiểm tra xem có phải là quy trình DDP hay không
        is_ddp = '.py' in sys.argv[0] and '_temp_' in sys.argv[0]
        if len(sys.argv) < 3 or is_ddp:
            print("DEBUG: Phát hiện quy trình DDP hoặc không đủ tham số")
            
            # Kiểm tra trong môi trường DDP có thể lấy lại đường dẫn cấu hình từ tham số gốc
            original_cfg_path = None
            try:
                # Đọc nội dung file DDP để tìm overrides
                if is_ddp:
                    with open(sys.argv[0], 'r') as f:
                        ddp_content = f.read()
                        if 'overrides =' in ddp_content:
                            # Tìm đường dẫn model trong overrides
                            model_line = [line for line in ddp_content.split('\n') if "'model':" in line or '"model":' in line]
                            if model_line:
                                import re
                                model_path = re.search(r"'model':\s*'([^']+)'|\"model\":\s*\"([^\"]+)\"", model_line[0])
                                if model_path:
                                    original_cfg_path = model_path.group(1) or model_path.group(2)
                                    print(f"DEBUG: Tìm thấy đường dẫn cấu hình từ DDP: {original_cfg_path}")
            except Exception as e:
                print(f"DEBUG: Không thể đọc file DDP: {e}")
                
            # Nếu tìm thấy đường dẫn cấu hình từ DDP, sử dụng nó
            if original_cfg_path and Path(original_cfg_path).exists():
                original_path = original_cfg_path
                print(f"DEBUG: Sử dụng cấu hình từ DDP: {original_path}")
            else:
                # Tìm các file yaml trong thư mục hiện tại
                yaml_files = [f for f in os.listdir('.') if f.endswith('.yaml')]
                # Ưu tiên file có DynamicHead trong tên
                dynamic_head_files = [f for f in yaml_files if 'DynamicHead' in f]
                
                if dynamic_head_files:
                    original_path = dynamic_head_files[0]
                    print(f"DEBUG: Tìm thấy file DynamicHead: {original_path}")
                elif yaml_files:
                    # Nếu không có file DynamicHead, thử kiểm tra nội dung các file YAML
                    for yaml_file in yaml_files:
                        try:
                            with open(yaml_file, 'r') as f:
                                if '"newhead": "DynamicHead"' in f.read() or "'newhead': 'DynamicHead'" in f.read():
                                    original_path = yaml_file
                                    print(f"DEBUG: Tìm thấy file có newhead=DynamicHead: {original_path}")
                                    break
                        except:
                            pass
                    else:  # Nếu không tìm thấy file có tham số newhead, lấy file đầu tiên
                        original_path = yaml_files[0]
                        print(f"DEBUG: Sử dụng file YAML đầu tiên: {original_path}")
                else:
                    # Kiểm tra biến môi trường
                    env_config = os.environ.get('ULTRALYTICS_CONFIG_PATH')
                    if env_config and os.path.exists(env_config):
                        original_path = env_config
                        print(f"DEBUG: Sử dụng cấu hình từ biến môi trường: {original_path}")
                    else:
                        # Sử dụng đường dẫn mặc định
                        original_path = "cfg/models/yolov8.yaml"
                        print(f"DEBUG: Sử dụng đường dẫn mặc định: {original_path}")
        else:
            # Chế độ bình thường, lấy từ tham số dòng lệnh
            original_path = sys.argv[2]
            print(f"DEBUG: Sử dụng đường dẫn từ tham số: {original_path}")
        
        print(f"DEBUG: original_path = {original_path}")
        new_path = original_path.replace("ultralytics\\", "").replace("\\", "/")
        print(f"DEBUG: new_path = {new_path}")

        FILE = Path(__file__).resolve()
        print(f"DEBUG: FILE = {FILE}")
        
        if current_os == "Windows":
            ROOT = FILE.parents[1]
        elif current_os == "Linux":
            ROOT = FILE.parents[2]
        else:
            ROOT = FILE.parents[2]
        print(f"DEBUG: ROOT = {ROOT}")
        
        DEFAULT_CFG_PATH = ROOT / new_path
        print(f"DEBUG: DEFAULT_CFG_PATH = {DEFAULT_CFG_PATH}")
        
        # Kiểm tra xem file có tồn tại không
        if not DEFAULT_CFG_PATH.exists():
            print(f"DEBUG: File {DEFAULT_CFG_PATH} không tồn tại!")
            
            # Thử tìm file trong thư mục hiện tại
            alt_path = Path(new_path)
            if alt_path.exists():
                print(f"DEBUG: Sử dụng đường dẫn thay thế: {alt_path}")
                DEFAULT_CFG_PATH = alt_path
            else:
                print("DEBUG: Không tìm thấy file cấu hình, sử dụng đối tượng mặc định")
                return IterableSimpleNamespace(newhead="")
            
        DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
        print(f"DEBUG: DEFAULT_CFG_DICT = {DEFAULT_CFG_DICT}")
        
        # # Đảm bảo luôn có tham số newhead
        # if 'newhead' not in DEFAULT_CFG_DICT:
        #     print("DEBUG: Thêm tham số newhead vào cấu hình")
        #     DEFAULT_CFG_DICT['newhead'] = "DynamicHead"
        
        DEFAULT_CFG_PA = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
        print(f"DEBUG: DEFAULT_CFG_PA = {DEFAULT_CFG_PA}")
        
        return DEFAULT_CFG_PA
    except Exception as e:
        print(f"DEBUG: Lỗi trong load_script(): {e}")
        # Trả về đối tượng mặc định trong trường hợp lỗi
        return IterableSimpleNamespace(newhead="DynamicHead")


# '''
#     @from MangoAI &3836712GKcH2717GhcK. please see https://github.com/iscyy/ultralyticsPro
# '''
# from ultralytics.utils import IterableSimpleNamespace
# from pathlib import Path
# import sys
# import os
# from ultralytics.utils import DEFAULT_CFG
# from ultralytics.cfg import get_cfg
# import platform
# current_os = platform.system()
# import yaml
# import re

# def yaml_load(file="data.yaml", append_filename=False):
#     print(f"DEBUG: yaml_load - Đang tải file: {file}")
#     assert Path(file).suffix in (".yaml", ".yml"), f"Attempting to load non-YAML file {file} with yaml_load()"
#     with open(file, errors="ignore", encoding="utf-8") as f:
#         s = f.read()  # string

#         if not s.isprintable():
#             s = re.sub(r"[^\x09\x0A\x0D\x20-\x7E\x85\xA0-\uD7FF\uE000-\uFFFD\U00010000-\U0010ffff]+", "", s)

#         data = yaml.safe_load(s) or {}  
#         if append_filename:
#             data["yaml_file"] = str(file)
#         print(f"DEBUG: yaml_load - Dữ liệu được tải: {data}")
#         return data

# def load_script():
#     print("DEBUG: Starting load_script()")
#     print(f"DEBUG: sys.argv = {sys.argv}")
    
#     try:
#         # Kiểm tra xem có phải là quy trình DDP hay không
#         is_ddp = '.py' in sys.argv[0] and '_temp_' in sys.argv[0]
#         if len(sys.argv) < 3 or is_ddp:
#             print("DEBUG: Phát hiện quy trình DDP hoặc không đủ tham số")
            
#             # Kiểm tra trong môi trường DDP có thể lấy lại đường dẫn cấu hình từ tham số gốc
#             original_cfg_path = None
#             try:
#                 # Đọc nội dung file DDP để tìm overrides
#                 if is_ddp:
#                     with open(sys.argv[0], 'r') as f:
#                         ddp_content = f.read()
#                         if 'overrides =' in ddp_content:
#                             # Tìm đường dẫn model trong overrides
#                             model_line = [line for line in ddp_content.split('\n') if "'model':" in line or '"model":' in line]
#                             if model_line:
#                                 import re
#                                 model_path = re.search(r"'model':\s*'([^']+)'|\"model\":\s*\"([^\"]+)\"", model_line[0])
#                                 if model_path:
#                                     original_cfg_path = model_path.group(1) or model_path.group(2)
#                                     print(f"DEBUG: Tìm thấy đường dẫn cấu hình từ DDP: {original_cfg_path}")
#             except Exception as e:
#                 print(f"DEBUG: Không thể đọc file DDP: {e}")
                
#             # Nếu tìm thấy đường dẫn cấu hình từ DDP, sử dụng nó
#             if original_cfg_path and Path(original_cfg_path).exists():
#                 original_path = original_cfg_path
#                 print(f"DEBUG: Sử dụng cấu hình từ DDP: {original_path}")
#             else:
#                 # Tìm các file yaml trong thư mục hiện tại
#                 yaml_files = [f for f in os.listdir('.') if f.endswith('.yaml')]
#                 # Ưu tiên file có DynamicHead trong tên
#                 dynamic_head_files = [f for f in yaml_files if 'DynamicHead' in f]
                
#                 if dynamic_head_files:
#                     original_path = dynamic_head_files[0]
#                     print(f"DEBUG: Tìm thấy file DynamicHead: {original_path}")
#                 elif yaml_files:
#                     # Nếu không có file DynamicHead, thử kiểm tra nội dung các file YAML
#                     for yaml_file in yaml_files:
#                         try:
#                             with open(yaml_file, 'r') as f:
#                                 if '"newhead": "DynamicHead"' in f.read() or "'newhead': 'DynamicHead'" in f.read():
#                                     original_path = yaml_file
#                                     print(f"DEBUG: Tìm thấy file có newhead=DynamicHead: {original_path}")
#                                     break
#                         except:
#                             pass
#                     else:  # Nếu không tìm thấy file có tham số newhead, lấy file đầu tiên
#                         original_path = yaml_files[0]
#                         print(f"DEBUG: Sử dụng file YAML đầu tiên: {original_path}")
#                 else:
#                     # Kiểm tra biến môi trường
#                     env_config = os.environ.get('ULTRALYTICS_CONFIG_PATH')
#                     if env_config and os.path.exists(env_config):
#                         original_path = env_config
#                         print(f"DEBUG: Sử dụng cấu hình từ biến môi trường: {original_path}")
#                     else:
#                         # Sử dụng đường dẫn mặc định
#                         original_path = "cfg/models/yolov8.yaml"
#                         print(f"DEBUG: Sử dụng đường dẫn mặc định: {original_path}")
#         else:
#             # Chế độ bình thường, lấy từ tham số dòng lệnh
#             original_path = sys.argv[2]
#             print(f"DEBUG: Sử dụng đường dẫn từ tham số: {original_path}")
        
#         print(f"DEBUG: original_path = {original_path}")
#         new_path = original_path.replace("ultralytics\\", "").replace("\\", "/")
#         print(f"DEBUG: new_path = {new_path}")

#         FILE = Path(__file__).resolve()
#         print(f"DEBUG: FILE = {FILE}")
        
#         if current_os == "Windows":
#             ROOT = FILE.parents[1]
#         elif current_os == "Linux":
#             ROOT = FILE.parents[2]
#         else:
#             ROOT = FILE.parents[2]
#         print(f"DEBUG: ROOT = {ROOT}")
        
#         DEFAULT_CFG_PATH = ROOT / new_path
#         print(f"DEBUG: DEFAULT_CFG_PATH = {DEFAULT_CFG_PATH}")
        
#         # Kiểm tra xem file có tồn tại không
#         if not DEFAULT_CFG_PATH.exists():
#             print(f"DEBUG: File {DEFAULT_CFG_PATH} không tồn tại!")
            
#             # Thử tìm file trong thư mục hiện tại
#             alt_path = Path(new_path)
#             if alt_path.exists():
#                 print(f"DEBUG: Sử dụng đường dẫn thay thế: {alt_path}")
#                 DEFAULT_CFG_PATH = alt_path
#             else:
#                 print("DEBUG: Không tìm thấy file cấu hình, sử dụng đối tượng mặc định")
#                 return IterableSimpleNamespace(newhead="")
            
#         DEFAULT_CFG_DICT = yaml_load(DEFAULT_CFG_PATH)
#         print(f"DEBUG: DEFAULT_CFG_DICT = {DEFAULT_CFG_DICT}")
        
#         # Đảm bảo luôn có tham số newhead
#         if 'newhead' not in DEFAULT_CFG_DICT:
#             print("DEBUG: Thêm tham số newhead vào cấu hình")
#             DEFAULT_CFG_DICT['newhead'] = "DynamicHead"
        
#         DEFAULT_CFG_PA = IterableSimpleNamespace(**DEFAULT_CFG_DICT)
#         print(f"DEBUG: DEFAULT_CFG_PA = {DEFAULT_CFG_PA}")
        
#         return DEFAULT_CFG_PA
#     except Exception as e:
#         print(f"DEBUG: Lỗi trong load_script(): {e}")
#         # Trả về đối tượng mặc định trong trường hợp lỗi
#         return IterableSimpleNamespace(newhead="DynamicHead")
