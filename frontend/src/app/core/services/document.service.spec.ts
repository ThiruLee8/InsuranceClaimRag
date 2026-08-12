import { TestBed } from '@angular/core/testing';
import { provideHttpClient } from '@angular/common/http';
import { provideHttpClientTesting, HttpTestingController } from '@angular/common/http/testing';
import { DocumentService } from './document.service';
import { environment } from '../../../environments/environment';

describe('DocumentService', () => {
  let service: DocumentService;
  let http: HttpTestingController;

  beforeEach(() => {
    TestBed.configureTestingModule({
      providers: [provideHttpClient(), provideHttpClientTesting()],
    });
    service = TestBed.inject(DocumentService);
    http = TestBed.inject(HttpTestingController);
  });

  afterEach(() => http.verify());

  it('lists documents', () => {
    service.list({ search: 'claim' }).subscribe((res) => {
      expect(res.total).toBe(1);
      expect(res.items[0].originalFileName).toBe('claim.pdf');
    });

    const req = http.expectOne(
      `${environment.apiBaseUrl}/documents?search=claim`
    );
    expect(req.request.method).toBe('GET');
    req.flush({
      success: true,
      message: 'OK',
      data: {
        total: 1,
        items: [{ id: '1', originalFileName: 'claim.pdf' }],
      },
    });
  });

  it('uploads a document', () => {
    const file = new File(['demo'], 'claim.txt', { type: 'text/plain' });
    service.upload(file).subscribe((doc) => {
      expect(doc.id).toBe('abc');
    });
    const req = http.expectOne(`${environment.apiBaseUrl}/documents/upload`);
    expect(req.request.method).toBe('POST');
    req.flush({ success: true, message: 'OK', data: { id: 'abc' } });
  });
});
