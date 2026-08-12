import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting, HttpTestingController } from '@angular/common/http/testing';
import { ConversationService } from './conversation.service';
import { environment } from '../../../environments/environment';

describe('ConversationService', () => {
  let service: ConversationService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(ConversationService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('creates a conversation', () => {
    service.create('Demo').subscribe((conv) => {
      expect(conv.title).toBe('Demo');
    });
    const req = http.expectOne(`${environment.apiBaseUrl}/conversations`);
    expect(req.request.method).toBe('POST');
    req.flush({
      success: true,
      message: 'OK',
      data: { id: '1', title: 'Demo' },
    });
  });
});
