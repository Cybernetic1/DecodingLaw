import scrapy
import sys
import requests

postLoginUrl='https://ids.hku.hk/idp/ProcessAuthnLib'
requestUrl='http://www.lexisnexis.com.eproxy1.lib.hku.hk/in/legal/api/version1/sr?sr='+sys.argv[1]+'&csi=305749,305740,305743,305744&oc=00240&shr=t&scl=t&hac=f&hct=f&nonLatin1Chars=true'

payload = {
	'userid':'username',
	'password':'password'
}
with requests.Session() as session:
	post = session.post(postLoginUrl,data=payload)
	r = session.get(requestUrl)
	
	class caseSpider(scrapy.Spider):
		name="lawCases"
		start_urls= [requestUrl,]

		def parse(self,response):