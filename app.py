import config
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel, Field
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from address_alignment import address_alignment
from transformers import BertForTokenClassification, BertTokenizerFast

# 加载模型
model_path = config.FINETUNED_PATH / "best"
model = BertForTokenClassification.from_pretrained(model_path).to(config.DEVICE)
tokenizer = BertTokenizerFast.from_pretrained(model_path)

# 创建 FastAPI 实例
app = FastAPI()

# 挂载静态文件
app.mount("/static", StaticFiles(directory="templates"), name="static")


class AddressAlignmentRequest(BaseModel):
    message: str = Field(..., example="地址文本信息")


class AddressAlignmentResponse(BaseModel):
    province: str | None = Field(..., example="省份")
    city: str | None = Field(..., example="城市")
    district: str | None = Field(..., example="区县")
    town: str | None = Field(..., example="乡/镇/街道")
    detail: str | None = Field(..., example="详细地址")


@app.get("/")
async def homepage():
    return FileResponse("templates/index.html")


@app.post("/address_alignment")
async def handle_message(request: AddressAlignmentRequest) -> AddressAlignmentResponse:
    user_message = request.message
    address = address_alignment(
        user_message, model, tokenizer, config.LABELS, config.MYSQL_CONFIG
    )
    res = AddressAlignmentResponse(
        province=address["省份"],
        city=address["城市"],
        district=address["区县"],
        town=address["街道"],
        detail=address["详细地址"],
    )
    return res


if __name__ == "__main__":
    # 启动服务
    uvicorn.run(app, host="0.0.0.0", port=8089)

# 测试数据：
#   中国浙江省杭州市余杭区葛墩路27号楼
#   北京市市辖区通州区永乐店镇27号楼
#   北京市市辖区东风街道27号楼
#   新疆维吾尔自治区划阿拉尔市金杨镇27号楼
#   甘肃省南市文县碧口镇27号楼
#   陕西省渭南市华阴市罗镇27号楼
#   西藏自治区拉萨市墨竹工卡县工卡镇27号楼
#   广州市花都区花东镇27号楼
