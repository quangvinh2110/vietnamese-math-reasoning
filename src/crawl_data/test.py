import requests
import json

url = "https://nlu-translate.vinai.io/translate"

payload = json.dumps({
  "mt_mode": "en2vi",
  "mt_input": "Of the 90 people on William's bus, 3/5 were Dutch. Of the 1/2 of the Dutch who were also American, 1/3 got window seats. What's the number of Dutch Americans who sat at the windows?"
})
headers = {
  'Accept': 'application/json, text/plain, */*',
  'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8,en;q=0.7',
  'Connection': 'keep-alive',
  'Content-Type': 'application/json',
  'Origin': 'https://vinai-translate.vinai.io',
  'Sec-Fetch-Dest': 'empty',
  'Sec-Fetch-Mode': 'cors',
  'Sec-Fetch-Site': 'same-site',
  'User-Agent': 'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
  'sec-ch-ua': '"Chromium";v="122", "Not(A:Brand";v="24", "Google Chrome";v="122"',
  'sec-ch-ua-mobile': '?0',
  'sec-ch-ua-platform': '"Linux"',
  'Cookie': 'TS01e39374=01dcc7e0b05336d78b3734f0dee5c36ea465956cb7339701f272e13b4003df2130b0e393fdba4272e884d80ca24ecdd737d4e2d38f'
}

response = requests.request("POST", url, headers=headers, data=payload)
response.encoding = response.apparent_encoding
print(response.json())
