import pytest
from playwright.sync_api import Page, expect

def test_chat_interface(page: Page):
    """Test chat interface functionality"""
    # Navigate to chat interface
    page.goto("http://localhost:3000")
    
    # Check initial rendering
    expect(page.locator("text=Tư Vấn Y Tế")).to_be_visible()
    expect(page.locator("textarea")).to_be_visible()
    expect(page.locator("button >> text=Send")).to_be_visible()

    # Test sending message
    page.fill("textarea", "Xin chào")
    page.click("button >> text=Send")
    
    # Wait for response
    expect(page.locator("text=Xin cho biết họ tên")).to_be_visible()

    # Test complete conversation flow
    test_inputs = [
        "Nguyễn Văn A",     # name
        "30",               # age
        "Nam",              # gender
        "test@test.com",    # contact
        "Đau đầu, sốt",    # symptoms
        "Không có",         # medical history
        "Có"               # booking confirmation
    ]

    for message in test_inputs:
        page.fill("textarea", message)
        page.click("button >> text=Send")
        # Wait for each response
        page.wait_for_timeout(1000)

    # Verify medical record generation
    expect(page.locator("text=Đã tạo hồ sơ khám bệnh")).to_be_visible()

def test_queue_status_display(page: Page):
    """Test queue status display"""
    page.goto("http://localhost:3000")
    
    # Complete conversation until booking
    # ... (similar to above)
    
    # Check queue status display
    expect(page.locator("text=Số thứ tự")).to_be_visible()
    expect(page.locator("text=Thời gian chờ")).to_be_visible()

def test_responsive_design(page: Page):
    """Test responsive design"""
    # Test mobile view
    page.set_viewport_size({"width": 375, "height": 667})
    page.goto("http://localhost:3000")
    expect(page.locator("textarea")).to_be_visible()

    # Test tablet view
    page.set_viewport_size({"width": 768, "height": 1024})
    expect(page.locator("textarea")).to_be_visible()

    # Test desktop view
    page.set_viewport_size({"width": 1920, "height": 1080})
    expect(page.locator("textarea")).to_be_visible()

def test_error_handling_display(page: Page):
    """Test error message display"""
    page.goto("http://localhost:3000")
    
    # Test invalid age input
    page.fill("textarea", "abc")  # invalid age
    page.click("button >> text=Send")
    expect(page.locator("text=Vui lòng nhập tuổi bằng số")).to_be_visible()

def test_vietnamese_text_display(page: Page):
    """Test Vietnamese text rendering"""
    page.goto("http://localhost:3000")
    
    # Test Vietnamese input
    vietnamese_text = "Tôi bị đau đầu"
    page.fill("textarea", vietnamese_text)
    page.click("button >> text=Send")
    
    # Verify text is displayed correctly
    expect(page.locator(f"text={vietnamese_text}")).to_be_visible()

def test_accessibility(page: Page):
    """Test accessibility features"""
    page.goto("http://localhost:3000")
    
    # Test keyboard navigation
    page.keyboard.press("Tab")
    expect(page.locator("textarea")).to_be_focused()
    
    # Test ARIA labels
    expect(page.locator("textarea[aria-label]")).to_be_visible()
    expect(page.locator("button[aria-label]")).to_be_visible()

@pytest.fixture(autouse=True)
def run_around_tests():
    """Setup and teardown for tests"""
    # Setup
    yield
    # Teardown