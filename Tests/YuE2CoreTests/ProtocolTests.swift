import XCTest
@testable import YuE2Core

final class ProtocolTests: XCTestCase {
    func testChunkRanges() throws {
        let r = try YuE2Protocol.chunkRanges(frames: 30000, prefixTokens: 1500)
        XCTAssertEqual(r.first!.count, (24576 - 1500 - 3) / 2)
        XCTAssertEqual(r.last!.upperBound, 30000)
    }
    func testRawT() {
        XCTAssertEqual(NARSolver.rawT(1.0), 20)
        XCTAssertEqual(NARSolver.rawT(0.5), 0, accuracy: 1e-12)
    }
}
