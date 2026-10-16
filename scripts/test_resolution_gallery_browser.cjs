// DOM-only browser checks: images are blocked and no screenshots are taken.
const {chromium} = require('playwright');
const assert = require('node:assert/strict');
const path = require('node:path');
const {pathToFileURL} = require('node:url');
(async () => {
  const browser = await chromium.launch({channel:"chromium",headless:true,args:['--blink-settings=imagesEnabled=false']});
  try {
    for (const url of [pathToFileURL(path.resolve(__dirname,'../index.html')).href,
                      process.env.RAPD_PREVIEW_URL || 'http://127.0.0.1:8765/']) {
      const page = await browser.newPage();
      const errors = [];
      page.on('pageerror',error => errors.push(error.message));
      await page.route('**/*',route => route.request().resourceType()==='image' ? route.abort() : route.continue());
      await page.goto(url);
      await page.locator('#gallery-pause').click();
      await page.locator('.gallery-item').first().click();
      const slider=page.locator('#resolution-slider');
      await page.waitForFunction(() => !document.getElementById('resolution-slider').disabled);
      const box=await slider.boundingBox();
      await page.mouse.move(box.x+box.width-8,box.y+box.height/2);
      await page.mouse.down();
      await page.mouse.move(box.x+box.width*0.3,box.y+box.height/2,{steps:12});
      await page.mouse.up();
      const value=Number(await slider.inputValue());
      assert(value>512 && value<4096,`Drag failed at ${url}: ${value}`);
      await slider.focus();
      await page.keyboard.press('Home');
      assert.equal(await slider.inputValue(),'512');
      await page.keyboard.press('ArrowRight');
      assert.equal(await slider.inputValue(),'576');
      await page.keyboard.press('End');
      assert.equal(await slider.inputValue(),'4096');
      await page.locator('#gallery-close').click();
      // Check every source and ensure reopening resets the slider correctly.
      for (let i=0;i<15;i++) {
        await page.locator('.gallery-item').nth(i).evaluate(button=>button.click());
        await page.waitForFunction(() => !document.getElementById('resolution-slider').disabled);
        assert.equal(await slider.inputValue(),'4096');
        await page.locator('#gallery-close').click();
      }
      assert.deepEqual(errors,[]);
      console.log(`${new URL(url).protocol}: drag, keyboard endpoints, and all 15 modal reopen checks passed`);
      await page.close();
    }
  } finally { await browser.close(); }
})().catch(error=>{console.error(error);process.exit(1);});
